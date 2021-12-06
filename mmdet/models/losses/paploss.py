import torch


# oracle loss
def paploss(logits, ious, targets, ctrl_points, num_topk, loss_form):
    pos_idx = (targets == 1)
    pos_logits = logits[pos_idx]
    pos_ious = ious[pos_idx]
    pos_num = len(pos_logits)

    valid_idx = (targets != -1)
    all_logits = logits[valid_idx]
    all_ious = ious[valid_idx]
    if all_logits.shape[0] > int(num_topk):
        topk_logits, topk_idx = torch.topk(all_logits, int(num_topk))
        topk_ious = all_ious[topk_idx]
    else:
        topk_logits = all_logits
        topk_ious = all_ious
    
    select_logits = topk_logits
    select_ious = topk_ious
    diff = select_logits[:, None] - pos_logits[None, :]  # [all_samples, pos_samples]
    norm_diff = diff / 2 + 0.5
    norm_diff = torch.clamp(norm_diff, min=0, max=1)
    
    if loss_form == 'searched':
        if ctrl_points.shape[0] == 4:
            H1 = torch.sign(diff) / 2 + 0.5
        elif ctrl_points.shape[0] == 5:
            H1 = _linear(norm_diff, ctrl_points[4])
            H1 = H1.detach()
        else:
            print('wrong shape!')

        H2 = _linear(norm_diff, ctrl_points[0])
        I2 = _linear(select_ious, ctrl_points[1])[:, None]
        I3 = _linear(pos_ious, ctrl_points[2])[None, :]
        I1 = _linear(pos_ious, ctrl_points[3])
    elif loss_form == 'sigmoid':
        H1 = torch.sigmoid(diff).detach()
        H2 = torch.sigmoid(diff)
        I2 = torch.sigmoid(select_ious * 2 - 1)[:, None]
        I3 = torch.sigmoid(pos_ious * 2 - 1)[None, :]
        I1 = torch.sigmoid(pos_ious * 2 - 1)
    elif loss_form == 'linear':
        H1 = norm_diff.detach()
        H2 = norm_diff
        I2 = select_ious[:, None]
        I3 = pos_ious[None, :]
        I1 = pos_ious
    elif loss_form == 'square':
        H1 = (norm_diff**2).detach()
        H2 = norm_diff**2
        I2 = (select_ious**2)[:, None]
        I3 = (pos_ious**2)[None, :]
        I1 = pos_ious**2

    loss = (1 - I1) + (H2 * (1 - I2) * I3).sum(0) / H1.sum(0)
    loss = loss.sum() / (pos_num + 1e-8)

    return loss


def _linear(x, ctrl_points):
    if ctrl_points.shape[0] in [8, 12, 16]:
        ctrl_x = ctrl_points[0::2]
        ctrl_y = ctrl_points[1::2]
        index = (x[..., None] > ctrl_x).sum(dim=-1) + (ctrl_x == 0).sum() * (x == 0) - 1
        y = (ctrl_y[index + 1] - ctrl_y[index]) / (ctrl_x[index + 1] - ctrl_x[index]) * (x - ctrl_x[index]) + ctrl_y[index]
    else:
        raise ValueError('wrong ctrl points')
    return y