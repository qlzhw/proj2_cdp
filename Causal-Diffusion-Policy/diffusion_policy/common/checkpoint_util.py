from typing import Optional, Dict
import os
import re

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
        self._load_existing_checkpoints()

    def _extract_monitor_value(self, filename):
        match = re.search(rf'{re.escape(self.monitor_key)}=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', filename)
        if match is None:
            return None
        return float(match.group(1))

    def _get_sorted_items(self):
        return sorted(self.path_value_map.items(), key=lambda x: x[1])

    def _load_existing_checkpoints(self):
        if self.k == 0 or (not os.path.exists(self.save_dir)):
            return

        for filename in os.listdir(self.save_dir):
            if (not filename.endswith('.ckpt')) or filename == 'latest.ckpt':
                continue
            value = self._extract_monitor_value(filename)
            if value is None:
                continue
            ckpt_path = os.path.join(self.save_dir, filename)
            self.path_value_map[ckpt_path] = value

        if len(self.path_value_map) <= self.k:
            return

        sorted_items = self._get_sorted_items()
        if self.mode == 'max':
            keep_items = sorted_items[-self.k:]
            remove_items = sorted_items[:-self.k]
        else:
            keep_items = sorted_items[:self.k]
            remove_items = sorted_items[self.k:]

        self.path_value_map = dict(keep_items)
        for ckpt_path, _ in remove_items:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    def get_metric_values(self):
        return [value for _, value in self._get_sorted_items()]

    def get_metric_mean(self):
        values = self.get_metric_values()
        if len(values) == 0:
            return None
        return sum(values) / len(values)
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = self._get_sorted_items()
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path
