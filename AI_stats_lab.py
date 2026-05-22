import numpy as np


# -------------------------------------------------
# Question 1: Joint Gaussian PDF and Marginals
# -------------------------------------------------

def joint_gaussian_pdf(x, y, mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6):
    q = (
        ((x - mu_x) ** 2) / (sigma_x ** 2)
        - 2 * rho *
        ((x - mu_x) * (y - mu_y)) /
        (sigma_x * sigma_y)
        + ((y - mu_y) ** 2) /
        (sigma_y ** 2)
    )

    coefficient = (
        1 / (
            2 *
            np.pi *
            sigma_x *
            sigma_y *
            np.sqrt(1 - rho ** 2)
        )
    )

    exponent = np.exp(
        -q / (2 * (1 - rho ** 2))
    )

    return coefficient * exponent


def marginal_pdf_x(x, mu_x=1, sigma_x=2):
    return (
        1 / (
            np.sqrt(2 * np.pi) *
            sigma_x
        )
    ) * np.exp(
        -((x - mu_x) ** 2) /
        (2 * sigma_x ** 2)
    )


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):
    return (
        1 / (
            np.sqrt(2 * np.pi) *
            sigma_y
        )
    ) * np.exp(
        -((y - mu_y) ** 2) /
        (2 * sigma_y ** 2)
    )


def covariance_matrix(sigma_x=2, sigma_y=3, rho=0.6):
    return np.array([
        [
            sigma_x ** 2,
            rho * sigma_x * sigma_y
        ],
        [
            rho * sigma_x * sigma_y,
            sigma_y ** 2
        ]
    ])


def joint_pdf_grid_integral(mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6, n=250):
    x_vals = np.linspace(
        mu_x - 4 * sigma_x,
        mu_x + 4 * sigma_x,
        n
    )

    y_vals = np.linspace(
        mu_y - 4 * sigma_y,
        mu_y + 4 * sigma_y,
        n
    )

    X, Y = np.meshgrid(
        x_vals,
        y_vals
    )

    Z = joint_gaussian_pdf(
        X,
        Y,
        mu_x,
        mu_y,
        sigma_x,
        sigma_y,
        rho
    )

    # NumPy 2.0+ compatible
    integral_x = np.trapezoid(
        Z,
        x_vals,
        axis=1
    )

    integral = np.trapezoid(
        integral_x,
        y_vals
    )

    return float(integral)


# -------------------------------------------------
# Question 2: Simulation and Independence
# -------------------------------------------------

def generate_joint_gaussian_samples(
    n=100000,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    seed=0
):
    np.random.seed(seed)

    mean = [
        mu_x,
        mu_y
    ]

    cov = covariance_matrix(
        sigma_x,
        sigma_y,
        rho
    )

    samples = np.random.multivariate_normal(
        mean,
        cov,
        size=n
    )

    x_samples = samples[:, 0]
    y_samples = samples[:, 1]

    return x_samples, y_samples


def sample_means(x_samples, y_samples):
    mean_x = np.mean(x_samples)
    mean_y = np.mean(y_samples)

    return mean_x, mean_y


def sample_covariance_matrix(x_samples, y_samples):
    return np.cov(
        x_samples,
        y_samples,
        ddof=1
    )


def sample_correlation(x_samples, y_samples):
    corr_matrix = np.corrcoef(
        x_samples,
        y_samples
    )

    return corr_matrix[0, 1]


def gaussian_independence_check(rho):
    return rho == 0


def zero_rho_covariance_check(n=100000):
    x_samples, y_samples = (
        generate_joint_gaussian_samples(
            n=n,
            rho=0,
            seed=42
        )
    )

    cov_matrix = (
        sample_covariance_matrix(
            x_samples,
            y_samples
        )
    )

    return bool(
        abs(cov_matrix[0, 1]) < 0.05
    )


def nonzero_rho_covariance_check(n=100000):
    sigma_x = 2
    sigma_y = 3
    rho = 0.6

    expected_cov = (
        rho *
        sigma_x *
        sigma_y
    )

    x_samples, y_samples = (
        generate_joint_gaussian_samples(
            n=n,
            rho=rho,
            seed=42
        )
    )

    cov_matrix = (
        sample_covariance_matrix(
            x_samples,
            y_samples
        )
    )

    return bool(
        abs(
            cov_matrix[0, 1] -
            expected_cov
        ) < 0.15
    )
