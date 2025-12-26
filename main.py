import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(cfg)
    print("latent_dim:", cfg.model.latent_dim)

if __name__ == "__main__":
    main()
