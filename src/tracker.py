from pathlib import Path
import numpy as np
import torch
import yaml


def create_botsort_tracker(tracker_config_path: Path, device: str = None):
    from boxmot.trackers.box.botsort.tracker import BotSort

    cfg_path = Path(tracker_config_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    if cfg.get('tracker_type', 'botsort').lower() != 'botsort':
        raise ValueError('Only botsort is supported')

    cmc = cfg.get('gmc_method', 'sof')
    cmc = {'sparseOptFlow': 'sof', 'sparseoptflow': 'sof'}.get(cmc, cmc)
    if cmc not in ('sof', 'ecc', 'orb', 'sift', 'none', None):
        raise ValueError(f'Unknown camera motion compensation method: {cmc!r}')

    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    reid_enabled = bool(cfg.get('with_reid', True))
    weights = Path(cfg.get('model', 'osnet_x0_25_msmt17.pt'))
    if not weights.is_absolute():
        weights = Path(__file__).resolve().parents[1] / 'models' / weights
    if reid_enabled and not weights.is_file():
        raise FileNotFoundError(f'ReID weights not found: {weights}')

    tracker = BotSort(
        track_high_thresh=float(cfg.get('track_high_thresh', 0.25)),
        track_low_thresh=float(cfg.get('track_low_thresh', 0.1)),
        new_track_thresh=float(cfg.get('new_track_thresh', 0.25)),
        track_buffer=int(cfg.get('track_buffer', 90)),
        match_thresh=float(cfg.get('match_thresh', 0.80)),
        proximity_thresh=float(cfg.get('proximity_thresh', 0.50)),
        appearance_thresh=float(cfg.get('appearance_thresh', 0.25)),
        use_cmc=cmc not in ('none', None),
        cmc_method=None if cmc == 'none' else cmc,
        use_embeddings=reid_enabled,
        reid_weights=str(weights) if reid_enabled else None,
        device=device,
        half=device.startswith('cuda'),
        per_class=False,
    )
    print(f'[FootballMOT] BotSort: device={device}, ReID={reid_enabled}, CMC={cmc}')
    return tracker


def parse_tracker_output(tracks):
    if tracks is None:
        return []
    arr = np.asarray(tracks)
    if not arr.size:
        return []
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    result = []
    for row in arr:
        if len(row) < 7:
            raise ValueError(f'Invalid BoxMOT row ({len(row)} columns): {row}')
        result.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]),
                       int(row[4]), float(row[5]), int(row[6])))
    return result
