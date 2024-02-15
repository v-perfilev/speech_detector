import os
import sys


class FileHandler:

    def get_file_paths(self, base_path, file_format, limit_per_folder=sys.maxsize):
        file_paths = []
        for root, dirs, files in os.walk(base_path):
            files_processed = 0
            for file_name in files:
                if file_name.endswith(f'.{file_format}'):
                    file_path = os.path.join(root, file_name)
                    file_paths.append(file_path)
                    files_processed += 1
                if files_processed >= limit_per_folder:
                    break
        return file_paths
