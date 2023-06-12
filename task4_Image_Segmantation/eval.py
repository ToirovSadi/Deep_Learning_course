import torch
import matplotlib.pyplot as plt
from utils import calc_metrics

def evaluate_plot(
    model,
    dataloader,
    loss,
    device,
    num_plots=6,
    plot=True,
):
    model.eval()
    history = evaluate(model, loss, dataloader, device)

    print("Summary:")
    for i in history:
        print(f"{i}: {history[i]}")
    
    if plot:
        dataset = dataloader.dataset
        if num_plots > len(dataset):
            num_plots = len(dataset)
        plt.figure(figsize=(num_plots * 2, 6))
        for i in range(num_plots):
            image = dataset[i][0].permute(1, 2, 0).numpy()
            ground_truth = dataset[i][1].permute(1, 2, 0).numpy()
            
            # predict image
            x = dataset[i][0].to(device)
            x = x.unsqueeze(0) # add dim=0, as batch_size
            x = model(x).cpu()
            logits = x.permute(0, 2, 3, 1).squeeze(0).detach()
            probs = torch.sigmoid(logits).numpy() # because model returns logits :)
            output_image = (probs > 0.5).astype(float)
            
            plt.subplot(3, num_plots, i + 1)
            plt.imshow(image)
            plt.title("Image")
            plt.axis("off")
            
            plt.subplot(3, num_plots, i + num_plots + 1)
            plt.imshow(ground_truth, cmap='gray')
            plt.title("Ground Truth")
            plt.axis("off")
            
            plt.subplot(3, num_plots, i + 2 * num_plots + 1)
            plt.imshow(output_image, cmap='gray')
            plt.title("Output")
            plt.axis("off")
        plt.show()


def evaluate(model, criterion, dataloader, device, other_metrics=True):
    model.eval()
    history = {}

    cur_loss = 0
    total_data = 0
    # i hope i can store them all in one tensor, since 
    # in val_dataset we should not have too much images
    all_targets = None
    all_outputs = None
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = criterion(output, y_batch)

        cur_loss += loss.item() * x_batch.size(0)
        total_data += x_batch.size(0)
        
        if other_metrics:
            all_targets = torch.cat([all_targets, y_batch], dim=0) if all_targets else y_batch 
            all_outputs = torch.cat([all_outputs, output], dim=0) if all_outputs else output

    loss = cur_loss / total_data
    history['loss'] = loss
    
    if other_metrics:
        # calculate metrics
        history.update(calc_metrics(all_outputs, all_targets))
    return history