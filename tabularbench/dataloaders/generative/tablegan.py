from tabularbench.dataloaders.generative.gan_file import GanFileDataLoader


class TableGanDataLoader(GanFileDataLoader):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            gan_name="tablegan",
            **kwargs,
        )
