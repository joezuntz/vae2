import numpy as np
from tensorflow import keras
from cosmosis.datablock import option_section

ZMIN = 0.005
ZMAX = 2.995
NZ = 300

class VAE:
    def __init__(self, directory, latent_size):
        self.model = keras.models.load_model(directory)
        self.latent_size = latent_size
        self.z = np.linspace(ZMIN, ZMAX, NZ)

    def generate(self, latent=None):
        if latent is None:
            latent = np.random.normal(size=(1, self.latent_size))
        output_data = self.model.decoder(latent)
        return self.z, output_data[0]


def setup(options):
    model_path = options.get_string(option_section, "model_path")
    latent_size = options.get_int(option_section, "latent_size")
    input_section = options.get_string(option_section, "input_section")
    output_section = options.get_string(option_section, "output_section")
    return {
        "model": VAE(model_path, latent_size),
        "input_section": input_section,
        "output_section": output_section
    }

def execute(block, config):
    vae = config["model"]
    input_section = config["input_section"]
    output_section = config["output_section"]
    latent = [block[input_section, f"latent_{i+1}"] for i in range(vae.latent_size)]

    # The input shape of the model is (1, latent_size)
    latent = np.array([latent])

    # Generate a new n(z) realization from the VAE
    z, nz = vae.generate(latent)
    block[output_section, "z"] = z

    # save everything to the block
    for i in range(vae.latent_size):
        nzi = nz[i] / np.trapz(nz[i], z)
        block[output_section, f"bin_{i+1}"] = nzi
    block[output_section, "nbin"] = vae.latent_size
    block[output_section, "nz"] = len(nzi)
    return 0
    


def main(path, latent_size):
    import matplotlib.pyplot as plt
    vae = VAE(path, latent_size)
    z, nz = vae.generate()
    for i in range(latent_size):
        plt.plot(z, nz[i], label=f'Bin {i}')
    plt.show()


if __name__ == '__main__':
    # Input
    import argparse
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('path', type=str, help='The path to the base folder')
    parser.add_argument("--latent_size", default=5, help="The latent dimension of the model")
    args = parser.parse_args()

    main(args.path, args.latent_size)