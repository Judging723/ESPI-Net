"""Paper-aligned ESPI-Net trainers and controlled ablation variants."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerHCMA import nnUNetTrainerHCMA


class nnUNetTrainerESPINet(nnUNetTrainerHCMA):
    """Full ESPI-Net: Input PAF + E0-E4 MSPF + TC (lambda=0.20)."""


class nnUNetTrainerESPINet_SingleV1(nnUNetTrainerESPINet):
    """Single-phase V1 spatial-backbone baseline."""

    use_input_fusion = False
    use_multiscale_fusion = False
    spatial_input_mode = "single"
    phase_index = 1
    temporal_weight = 0.0


class nnUNetTrainerESPINet_Concat(nnUNetTrainerESPINet):
    """Unweighted V0-V3 channel-concatenation baseline."""

    use_input_fusion = False
    use_multiscale_fusion = False
    spatial_input_mode = "concat"
    temporal_weight = 0.0


class nnUNetTrainerESPINet_InputPAF(nnUNetTrainerESPINet):
    """Input PAF without encoder-stage MSPF or TC."""

    use_input_fusion = True
    use_multiscale_fusion = False
    temporal_weight = 0.0


class nnUNetTrainerESPINet_MSPF(nnUNetTrainerESPINet):
    """Concatenated V0-V3 input plus E0-E4 MSPF, without TC."""

    use_input_fusion = False
    use_multiscale_fusion = True
    spatial_input_mode = "concat"
    temporal_weight = 0.0


class nnUNetTrainerESPINet_NoTC(nnUNetTrainerESPINet):
    """Input PAF + E0-E4 MSPF, without TC."""

    temporal_weight = 0.0


class nnUNetTrainerESPINet_E4(nnUNetTrainerESPINet):
    fusion_levels = (4,)


class nnUNetTrainerESPINet_E3E4(nnUNetTrainerESPINet):
    fusion_levels = (3, 4)


class nnUNetTrainerESPINet_E2E4(nnUNetTrainerESPINet):
    fusion_levels = (2, 3, 4)


class nnUNetTrainerESPINet_E1E4(nnUNetTrainerESPINet):
    fusion_levels = (1, 2, 3, 4)

