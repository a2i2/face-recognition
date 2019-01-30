import os


def find_duplicate_filenames(manifest_file):
    """
    Searches through a manifest file and returns a report of all duplicate filenames.
    The format of the report is as follows:

    {
        "file.txt": [
            "/path/to/first/instance/of/file.txt",
            "/path/to/second/instance/of/file.txt",
            "/path/to/third/instance/of/file.txt"
        ]
    }
    """
    files = dict()
    duplicates = dict()

    with open(manifest_file) as manifest:
        for line in manifest:
            line = line.strip()
            path, filename = os.path.split(line)

            if filename not in files:
                files[filename] = []

            files[filename].append(line)

    for filename, paths in files.items():
        if len(paths) > 1:
            duplicates[filename] = paths

    return duplicates
