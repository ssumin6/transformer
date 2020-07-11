import os
import argparse
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen
from model import Encoder, Decoder

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def is_finished(pred, max_length, eos_idx):
    if (pred.size()[1]>=max_length):
        return True
    cnt = 0
    pred = pred.tolist()
    for line in pred:
        if eos_idx in line:
            cnt += 1
    if (cnt == len(pred)):
        return True
    else:
        return False


def main(args):
    src, tgt = load_data(args.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    N = 6
    dim = 512

    # MODEL Construction
    encoder = Encoder(N, dim, pad_idx, src_vocab_size, device).to(device)
    decoder = Decoder(N, dim, pad_idx, tgt_vocab_size, device).to(device)

    if args.model_load:
        ckpt = torch.load("drive/My Drive/checkpoint/best.ckpt")
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])

    params = list(encoder.parameters())+list(decoder.parameters())

    if not args.test:
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        warmup = 4000
        steps = 1
        lr = 1.*(dim**-0.5)*min(steps**-0.5, steps*(warmup**-1.5))
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.98), eps=1e-09)

        train_losses = []
        val_losses = []
        latest = 1e08 # to store latest checkpoint

        start_epoch = 0

        if (args.model_load):
            start_epoch = ckpt["epoch"]
            optimizer.load_state_dict(ckpt["optim"])
            steps = start_epoch * 30

        for epoch in range(start_epoch, args.epochs):
            
            for src_batch, tgt_batch in train_loader:
                encoder.train()
                decoder.train()
                optimizer.zero_grad()
                tgt_batch = torch.LongTensor(tgt_batch)

                src_batch = Variable(torch.LongTensor(src_batch)).to(device)
                gt = Variable(tgt_batch[:, 1:]).to(device)  
                tgt_batch = Variable(tgt_batch[:, :-1]).to(device)

                enc_output, seq_mask = encoder(src_batch)
                dec_output = decoder(tgt_batch, enc_output, seq_mask)

                gt = gt.view(-1)
                dec_output = dec_output.view(gt.size()[0],-1)

                loss = F.cross_entropy(dec_output, gt, ignore_index=pad_idx)
                loss.backward()
                train_losses.append(loss.item())
                optimizer.step()

                steps += 1
                lr = (dim**-0.5)*min(steps**-0.5, steps*(warmup**-1.5))
                update_lr(optimizer, lr)

                if (steps%10 ==0):
                    print("loss : %f" % loss.item())
            
            for src_batch, tgt_batch in valid_loader:
                encoder.eval()
                decoder.eval()

                src_batch = Variable(torch.LongTensor(src_batch)).to(device)
                tgt_batch = torch.LongTensor(tgt_batch)
                gt = Variable(tgt_batch[:, 1:]).to(device)
                tgt_batch = Variable(tgt_batch[:, :-1]).to(device)

                enc_output, seq_mask = encoder(src_batch)
                dec_output = decoder(tgt_batch, enc_output, seq_mask)

                gt = gt.view(-1)
                dec_output = dec_output.view(gt.size()[0],-1)

                loss = F.cross_entropy(dec_output, gt, ignore_index=pad_idx)

                val_losses.append(loss.item())
            print("[EPOCH %d] Loss %f" %(epoch, loss.item()))

            if (val_losses[-1]<=latest):
                checkpoint = {'encoder':encoder.state_dict(), 'decoder':decoder.state_dict(), \
                    'optim':optimizer.state_dict(), 'epoch':epoch}
                torch.save(checkpoint, "drive/My Drive/checkpoint/best.ckpt")
                latest = val_losses[-1]

            if (epoch % 20 == 0):
                plt.figure()
                plt.plot(val_losses)
                plt.xlabel("epoch")
                plt.ylabel("model loss")
                plt.show()

    else:
        # test
        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        # LOAD CHECKPOINT

        pred = []
        for src_batch, tgt_batch in test_loader:
            encoder.eval()
            decoder.eval()

            b_s = min(args.batch_size, len(src_batch))
            tgt_batch = torch.zeros(b_s, 1).to(device).long()
            src_batch = Variable(torch.LongTensor(src_batch)).to(device)

            enc_output, seq_mask = encoder(src_batch)
            pred_batch = decoder(tgt_batch, enc_output, seq_mask)
            _, pred_batch = torch.max(pred_batch, 2)

            while (not is_finished(pred_batch, max_length, eos_idx)):
                # do something
                next_input = torch.cat((tgt_batch, pred_batch.long()), 1)
                pred_batch = decoder(next_input, enc_output, seq_mask)
                _, pred_batch = torch.max(pred_batch, 2)
            # every sentences in pred_batch should start with <sos> token (index: 0) and end with <eos> token (index: 1).
            # every <pad> token (index: 2) should be located after <eos> token (index: 1).
            # example of pred_batch:
            # [[0, 5, 6, 7, 1],
            #  [0, 4, 9, 1, 2],
            #  [0, 6, 1, 2, 2]]
            pred_batch = pred_batch.tolist()
            for line in pred_batch:
                line[-1] = 1
            pred += seq2sen(pred_batch, tgt_vocab)
            # print(pred)
        
        with open('results/pred.txt', 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))

        os.system('bash scripts/bleu.sh results/pred.txt multi30k/test.de.atok')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument(
        '--path',
        type=str,
        default='multi30k')

    parser.add_argument(
        '--epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--test',
        action='store_true')
    parser.add_argument(
        '--model_load',
        action='store_true')
    args = parser.parse_args()

    main(args)
