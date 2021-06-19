import resource
from time import sleep

# Measure interval in seconds
INTERVAL = 0.01

class MemoryMonitor:
    def __init__(self):
        self.keep_measuring = True

    def measure_usage(self):
        max_usage = 0
        while self.keep_measuring:
            # RUSAGE_SELF: request resources consumed by the calling process,
            # which is the sum of resources used by all threads in the process
            # ru_maxrss: This is the maximum resident set size used (in kilobytes)
            # https://manpages.debian.org/buster/manpages-dev/getrusage.2.en.html
            max_usage = max(
                max_usage,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            sleep(INTERVAL)

        return max_usage