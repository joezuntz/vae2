from tjpcov.covariance_calculator import CovarianceCalculator
import sacc
import numpy as np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute the covariance matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("INPUT", type=str, help="Input YAML data file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="summary_statistics.sacc",
        help="Output file name",
    )
    args = parser.parse_args()

    cov = CovarianceCalculator(args.INPUT)
    cov.create_sacc_cov(args.output)


    s = sacc.Sacc.load_fits(args.output)
    C = s.covariance.dense
    try:
        np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        print("Covariance matrix is not positive definite")
        d = C.diagonal()
        print(d.min())
        C[np.diag_indices_from(C)] *= (1 + 1e-4)
        np.linalg.cholesky(C)
        s.add_covariance(C, overwrite=True)
        s.save_fits(args.output, overwrite=True)