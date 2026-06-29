from pyscf.qr.hf import QR


class UQR(QR):
    '''Quadratic response for unrestricted UHF/UKS references.'''

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Unrestricted QR is not implemented.')

    def kernel(self, *args, **kwargs):
        raise NotImplementedError('Unrestricted QR is not implemented.')

    def get_2tdm(self, i, j):
        raise NotImplementedError('Unrestricted QR is not implemented.')
