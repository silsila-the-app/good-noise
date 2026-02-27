from .pgd_embedding import PGDEmbeddingAttack
from .psychoacoustic_pgd import PsychoacousticPGDAttack
from .mel_disruption import MelDisruptionAttack
from .universal_pgd import UniversalPGDAttack
from .asr_disruption import ASRDisruptionAttack
from .spectral_filter import SpectralFilterAttack
from .combine import CombineAttack

ATTACK_REGISTRY = {
    "pgd_embedding": PGDEmbeddingAttack,
    "psychoacoustic_pgd": PsychoacousticPGDAttack,
    "mel_disruption": MelDisruptionAttack,
    "universal_pgd": UniversalPGDAttack,
    "asr_disruption": ASRDisruptionAttack,
    "spectral_filter": SpectralFilterAttack,
    "combine": CombineAttack,
}

__all__ = list(ATTACK_REGISTRY.keys()) + ["ATTACK_REGISTRY"]
