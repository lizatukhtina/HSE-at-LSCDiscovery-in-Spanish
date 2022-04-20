import os
import random
import re
from argument_parser import CustomArgumentParser, Argument, SemicolonSeparatedArgument, EnumArgument
from typing import List


class BasePreprocessor:
    def __init__(self, targets_path: str, corpora_paths: List[str], output_folder: str):
        with open(targets_path) as f:
            self.targets = [i.strip() for i in f.readlines()]
        self.corpora_paths = corpora_paths
        self.output_folder = output_folder

    def process_line(self, line: str) -> list:
        return line.split()

    def process_corpora(self) -> None:
        all_lines = []
        os.makedirs(self.output_folder, exist_ok=True)
        for file in self.corpora_paths:
            with open(file, encoding='utf-8-sig') as f_in:
                print(f'Processing file {file}')
                lines = []
                for line in f_in:
                    processed_line = self.process_line(line)
                    if processed_line:
                        chunks = [processed_line[i:i + 510] for i in range(0, len(processed_line), 510)]
                        # BERT has a limit of 512 tokens, we reserve 2 tokens for ['SEP'] and ['CLS']
                        lines.extend([' '.join(chunk) for chunk in chunks])
                with open(os.path.join(self.output_folder, f'processed_{os.path.basename(file)}'), 'w',
                          encoding='utf-8') as f_out:
                    f_out.writelines([f'{line}\n' for line in lines])
                all_lines.extend(lines)
        self.generate_train_test(all_lines)

    def generate_train_test(self, lines: list) -> None:
        random.shuffle(lines)
        train_cutoff = int(len(lines) * 0.9)
        with open(os.path.join(self.output_folder, 'train.txt'), 'w', encoding='utf-8') as f:
            f.writelines([f'{line}\n' for line in lines[:train_cutoff]])
        with open(os.path.join(self.output_folder, 'test.txt'), 'w', encoding='utf-8') as f:
            f.writelines([f'{line}\n' for line in lines[train_cutoff:]])


class EnglishPreprocessor(BasePreprocessor):
    def __init__(self, targets_path: str, corpora_paths: List[str], output_folder: str):
        super().__init__(targets_path, corpora_paths, output_folder)

    def process_line(self, line: str) -> list:
        words = line.split()
        for target in self.targets:
            bare_target = target[:-3]  # stripping part of speech tag
            if bare_target in words:
                break  # it means that the line contains a word that is identical to a target word,
                # but has a different part of speech. We drop such lines for simplicity.
            if target in words:
                words = [i if i != target else bare_target for i in words]
                # we substitute target words with their versions without a POS tag
        else:  # if break didn't trigger, we add such a line to the result
            return words


class LatinPreprocessor(BasePreprocessor):
    def __init__(self, targets_path: str, corpora_paths: List[str], output_folder: str):
        super().__init__(targets_path, corpora_paths, output_folder)

    def process_line(self, line: str) -> list:
        clean_line = re.sub(r'#\d', '', line)
        return clean_line.split()


arg_list = [Argument(name='targets_path', type_=str, help_message='Path to a .txt file with target words',
                     absent_message='Please specify a path to target words'),
            SemicolonSeparatedArgument(name='corpora_paths', type_=str,
                                       help_message='Paths to corpora separated with a semicolon',
                                       absent_message='Please specify paths to corpora separated with a semicolon'),
            EnumArgument(name='corpora_language', type_=str, help_message='Corpora language',
                         absent_message='Please specify a valid corpora language',
                         choices=list(['english', 'spanish', 'german', 'latin', 'swedish'])),
            Argument(name='output_folder', type_=str, help_message='Path to a folder for processed files',
                     absent_message='Please specify output folder')]
arg_parser = CustomArgumentParser()
args = arg_parser.parse(arg_list)
if args.corpora_language == 'english':
    preprocessor = EnglishPreprocessor(targets_path=args.targets_path, corpora_paths=args.corpora_paths.split(';'),
                                       output_folder=args.output_folder)
elif args.corpora_language == 'latin':
    preprocessor = LatinPreprocessor(targets_path=args.targets_path, corpora_paths=args.corpora_paths.split(';'),
                                     output_folder=args.output_folder)
else:
    preprocessor = BasePreprocessor(targets_path=args.targets_path, corpora_paths=args.corpora_paths.split(';'),
                                    output_folder=args.output_folder)
preprocessor.process_corpora()
