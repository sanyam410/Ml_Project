import sys
from src.logger import logging


def error_message(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    error_msg = f"Error occurred in python script name [{exc_tb.tb_frame.f_code.co_filename}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_msg


class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(error)
        self.error_message = error_message(error, error_detail=error_detail)

    def __str__(self):
        return self.error_message

