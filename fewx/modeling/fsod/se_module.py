from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel // 2, bias=False),
            nn.Sigmoid()
        )

    def detectron_weight_mapping(self):
        detectron_weight_mapping={
            'fc.0.weight': 'fc.0.w',
            'fc.2.weight': 'fc.2.w',
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, concated, query):
        b, c, _, _ = concated.size()
        concated = concated.view(b, c)  # remove (1,1)
        attention = self.fc(concated).view(b, c//2, 1, 1)
        return query * attention.expand_as(query)
