"""Monitor log file for BENCHMARK DEBUG entries in real-time"""
import time
import sys

log_file = "logs/vp_investments.log"
print(f"Monitoring {log_file} for [BENCHMARK DEBUG] entries...")
print("Press Ctrl+C to stop\n")

# Get current file size to start reading from end
try:
    with open(log_file, 'r', encoding='utf-8') as f:
        f.seek(0, 2)  # Seek to end
        current_pos = f.tell()
except FileNotFoundError:
    print(f"Error: {log_file} not found")
    sys.exit(1)

try:
    while True:
        with open(log_file, 'r', encoding='utf-8') as f:
            f.seek(current_pos)
            new_lines = f.readlines()
            current_pos = f.tell()
            
            for line in new_lines:
                if '[BENCHMARK DEBUG]' in line or 'PHASE 6:' in line or 'PHASE 7:' in line:
                    print(line.strip())
        
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")
