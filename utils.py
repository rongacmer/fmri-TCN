import sys

def print_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    sys.stdout.flush()


def logger(info):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % 1 == 0:
        train_acc, test_acc, val_acc, val_loss = info['train_acc'], info['test_acc'], info['val_acc'],info['val_loss']
        time_test_acc, time_val_acc = info['time_test_accs'], info['time_val_accs']
        print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f} Val Accuracy: {:.3f} Val Loss: {:.3f} '
              'Time Test Accuracy: {:.3f} Time Val Accuracy:{:.3f}'.format(
            fold, epoch, train_acc, test_acc, val_acc,val_loss, time_test_acc, time_val_acc))
    sys.stdout.flush()


