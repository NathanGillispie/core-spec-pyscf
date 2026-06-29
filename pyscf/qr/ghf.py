from pyscf.qr.hf import QR


class GQR(QR):
    '''Quadratic response for generalized GHF/GKS references.'''

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Generalized QR is not implemented.')

    def kernel(self, *args, **kwargs):
        raise NotImplementedError('Generalized QR is not implemented.')

    def get_2tdm(self, i, j):
        raise NotImplementedError('Generalized QR is not implemented.')
