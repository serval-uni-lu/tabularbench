from tabularbench.dataloaders.generative.gan_file import GanFileDataLoader


class GoggleDataLoader(GanFileDataLoader):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            gan_name="goggle",
            **kwargs,
        )
