from tabularbench.dataloaders.generative.gan_file import GanFileDataLoader


class WGanDataLoader(GanFileDataLoader):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            gan_name="wgan",
            **kwargs,
        )
