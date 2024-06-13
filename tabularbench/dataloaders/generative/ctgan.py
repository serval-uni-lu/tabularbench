from tabularbench.dataloaders.generative.gan_file import GanFileDataLoader


class CTGanDataLoader(GanFileDataLoader):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            gan_name="ctgan",
            **kwargs,
        )
