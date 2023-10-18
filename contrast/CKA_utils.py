import torch

# Define the CCA loss function
def cca_loss(x, y):
    # Normalize the input data
    x_normalized = torch.nn.functional.normalize(x, dim=0)
    y_normalized = torch.nn.functional.normalize(y, dim=0)
    # Compute the covariance matrix of the normalized input data
    cov = torch.matmul(x_normalized.T, y_normalized)
    # Compute the singular value decomposition of the covariance matrix
    u, s, v = torch.svd(cov)
    # Compute the canonical correlation coefficients
    cca_coef = s[:min(x.shape[1], y.shape[1])]
    # Normalize the CCA coefficients
    cca_coef_norm = cca_coef / torch.max(cca_coef)
    # Compute the loss function
    loss = 1 - cca_coef_norm.sum()
    return loss


# Define the CKA loss function
def cka_loss(x, y):
    # Compute the Gram matrix of the input data
    x_gram = torch.matmul(x, x.t())
    y_gram = torch.matmul(y, y.t())

    # Compute the normalization factors for the Gram matrices
    x_norm = torch.norm(x_gram, p='fro')
    y_norm = torch.norm(y_gram, p='fro')

    # Compute the centered Gram matrix of the input data
    x_centered = x_gram - torch.mean(x_gram, dim=1, keepdim=True) - torch.mean(x_gram, dim=0, keepdim=True) + torch.mean(x_gram)
    y_centered = y_gram - torch.mean(y_gram, dim=1, keepdim=True) - torch.mean(y_gram, dim=0, keepdim=True) + torch.mean(y_gram)

    # Compute the normalization factors for the centered Gram matrices
    x_centered_norm = torch.norm(x_centered, p='fro')
    y_centered_norm = torch.norm(y_centered, p='fro')

    # Compute the centered kernel alignment between the input data
    cka = torch.trace(torch.matmul(x_centered, y_centered)) / (x_centered_norm * y_centered_norm)

    # Compute the loss function
    loss = 1 - cka
    return loss


def rbf(X, sigma=None):
    # X = torch.tensor(X)
    GX = torch.matmul(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = torch.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX

def kernel_HSIC(X, Y, gamma):
    n1 = X.shape[0]
    n2 = Y.shape[0]
    H1 = torch.eye(n1) - torch.ones(n1, n1) / n1
    H2 = torch.eye(n2) - torch.ones(n2, n2) / n2
    K1 = torch.matmul(torch.matmul(H1, rbf(X, gamma)), H1)
    K2 = torch.matmul(torch.matmul(H2, rbf(Y, gamma)), H2)
    hsic = torch.trace(torch.matmul(K1, K2))
    return hsic

