from eval import evaluate

def train_epoch(model, criterion, optimizer, dataloader, device):
    model.train()

    cur_loss = 0
    total_data = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        cur_loss += loss.item() * x_batch.size(0)
        total_data += x_batch.size(0)

    epoch_loss = cur_loss / total_data
    return epoch_loss


def train(model, epochs, criterion, optimizer, device, dataloaders, lr_scheduler=None, verbose=True):
  history = {}
  model.to(device)
  for epoch in range(epochs):
    train_loss = train_epoch(model, criterion, optimizer, dataloaders['train'], device)
    eval_hist = evaluate(model, criterion, dataloaders['val'], device)
    
    val_loss = eval_hist['loss']
    val_iou = eval_hist['jaccard_index']
    val_dice = eval_hist['dice_coef']
    
    log = []
    log.append(f"Epoch {epoch+1:03d}/{epochs:03d}")
    log.append(f"train_loss: {train_loss:0.5f}")
    log.append(f"val_loss: {val_loss:0.5f}")
    log.append(f"val_iou: {val_iou:0.5f}")
    log.append(f"val_dice: {val_dice:0.5f}")
    
    history['train_loss'] = history.get('train_loss', []) + [train_loss]
    history['val_loss'] = history.get('val_loss', []) + [val_loss]
    history['val_iou'] = history.get('val_iou', []) + [val_iou]
    history['val_dice'] = history.get('val_dice', []) + [val_dice]

    if verbose:
        print(" | ".join(log))
    
    if lr_scheduler is not None:
      lr_scheduler.step(val_loss)
      history['lr'] = history.get('lr', []) + [lr_scheduler._last_lr]

  return history
