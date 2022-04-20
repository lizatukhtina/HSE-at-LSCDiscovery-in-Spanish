import argparse
from typing import List, Union


class Argument:
    def __init__(self, name: str, type_: Union[type, None], help_message: str, absent_message: str):
        self.name = name
        self.key = f'--{name}'
        self.type_ = type_
        self.help_message = help_message
        self.absent_message = f'{absent_message} ({self.key})'
        self.action = None

    def validate(self, value):
        if not value:
            print(self.absent_message)
            exit(1)


class EnumArgument(Argument):
    def __init__(self, name: str, type_: Union[type, None], help_message: str, absent_message: str, choices: list):
        super().__init__(name=name, type_=type_, help_message=help_message, absent_message=absent_message)
        self.choices = choices
        self.choices_message = ", ".join(choices)
        self.help_message = f'{help_message} ({self.choices_message})'
        self.absent_message = f'{absent_message} ({self.choices_message}) ({self.key})'

    def validate(self, value):
        super().validate(value)
        if value not in self.choices:
            print(f'Invalid value for {self.key}. Must be one of these: {self.choices_message}')
            exit(1)


class SemicolonSeparatedArgument(Argument):
    def __init__(self, name: str, type_: Union[type, None], help_message: str, absent_message: str):
        super().__init__(name=name, type_=type_, help_message=help_message, absent_message=absent_message)

    def validate(self, value):
        super().validate(value)
        if value.count(';') != 1:
            print(f'Values for {self.key} must be separated with a single semicolon')
            exit(1)


class OptionalArgument(Argument):
    def __init__(self, name: str, type_: Union[type, None], help_message: str):
        super().__init__(name=name, type_=type_, help_message=help_message, absent_message='')

    def validate(self, value):
        pass


class BooleanArgument(Argument):
    def __init__(self, name: str, help_message: str, action: str):
        super().__init__(name=name, type_=None, help_message=help_message, absent_message='')
        self.action = action

    def validate(self, value):
        pass


class CustomArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

    def parse(self, args_unparsed: List[Argument]):
        for argument in args_unparsed:
            if argument.action:
                self.add_argument(argument.key, action=argument.action, help=argument.help_message)
            else:
                self.add_argument(argument.key, type=argument.type_, help=argument.help_message)
        args_parsed = self.parse_args()
        for unparsed in args_unparsed:
            unparsed.validate(getattr(args_parsed, unparsed.name))
        return args_parsed
