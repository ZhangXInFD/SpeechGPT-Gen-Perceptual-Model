from .transformer import (
    TransformerEncoder,
    UformerEncoder,
    Transformer,
    Uformer
)

from .conformer import (
    Conformer,
    UConformer
)

NET_NAME_DICT = {'transformer':TransformerEncoder, 'uconformer': UConformer, 'conformer': Conformer, 'uformer': UformerEncoder}

