import torch

def get_n_params(model):
    trainable_params = 0
    not_trainable_params = 0
    total_params = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        if p.requires_grad:
            trainable_params += nn
        else:
            not_trainable_params += nn
        total_params += nn
    return trainable_params, not_trainable_params, total_params

def build_optimizer(model, learning_rate, weight_decay):

    num_train, num_freeze, num_tot = get_n_params(model)
    print(f'# of trainable params : {num_train}')
    print(f'# of freezed params : {num_freeze}')
    print(f'# of total params : {num_tot}')
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    print(f'len trainable : {len(trainable_params)}')

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return optimizer
