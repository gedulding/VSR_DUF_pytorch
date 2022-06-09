import tqdm
import torch
from PIL import Image
import numpy as np
from tensorboardX import SummaryWriter
from utils import HuberLoss, loadyaml

def testing_model(model, test_data_loader , test_label_loader):
    model.eval()
    total_test_loss = 0
    yamfile = './config/config.yaml'
    config = loadyaml(yamfile)
    batch_size = float(config['test']['batch_size'])
    writer = SummaryWriter('runs/test')
    with torch.no_grad():
        data_loader = tqdm.tqdm(test_data_loader, desc='Test data loader')
        label_loader = tqdm.tqdm(test_label_loader, desc='Test label loader')
        for i , (data, target) in enumerate(zip(data_loader, label_loader)):
            target = target[0].cuda(); data = data[0].cuda()
            output = model.inference(data)                     # B*C*H*W

            for j in range(output.shape[0]):
                output = np.clip(output, 0, 1)
                Image.fromarray(np.around(output[0, 0] * 255).astype(np.uint8)).save(
                    './results/Frame{:03d}.png'.format(i*output.shape[0] + j))

            total_test_loss += HuberLoss(output, target)  # Huber loss

        total_test_loss /= (len(data_loader.dataset) / batch_size)

        writer.add_scalar('test_loss', total_test_loss)
