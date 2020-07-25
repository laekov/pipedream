import json
import time


class Recorder:
    def __init__(self, enabled, tid):
        self.enabled = enabled
        if not enabled:
            self.start_event = self.dummy
            self.end_event = self.dummy
        self.events = []
        self.tid = tid

    def dummy(self):
        pass
    
    def start_event(self):
        if not self.enabled:
            return
        self.timestamp = time.time_ns()

    def end_event(self, event, args={}):
        if not self.enabled:
            return
        current = time.time_ns()
        self.events.append(dict(
            name=event,
            ph='X',
            ts=self.timestamp,
            dur=current - self.timestamp,
            tid=self.tid,
            pid='Training',
            id=len(self.events),
            args=args
        ))
        self.timestamp = current

    def dump(self, filename):
        if not self.enabled:
            return
        with open(filename, 'w') as f:
            json.dump(self.events, f)

