import bookutils
from typing import List, Tuple, Dict, Any
from Fuzzer import RandomFuzzer
from html.parser import HTMLParser
from Coverage import Coverage
import pickle
import hashlib

# Create simple program-under-test
def html_parser(inp: str) -> None:
    parser = HTMLParser()
    parser.feed(inp)

def getTraceHash(cov: Coverage) -> str:
    pickledCov = pickle.dumps(cov.coverage())
    hashedCov = hashlib.md5(pickledCov).hexdigest()
    return hashedCov

def get_coverage(PUT, inp):
    with Coverage() as cov:
        try:
            PUT(inp)
        except BaseException:
            pass
    return cov

def rand_hexstr(debug: bool=False) -> str:
    fuzzer = RandomFuzzer(
        min_length=1, max_length=100, char_start=32, char_range=94
    )
    if debug:
        print("Initialized the fuzzer (random string generator).")
    random_input = fuzzer.fuzz()
    if debug:
        print(f"Generated random input: {random_input}")
    PUT = html_parser
    cov = get_coverage(PUT, random_input)
    if debug:
        print("Covered lines:")
        covered_lines = cov.coverage()
        for lines in list(covered_lines):
            print("| ", lines)
    cov_hash = getTraceHash(cov)
    if debug:
        print(f"Hashed coverage: {cov_hash}")
    return cov_hash
