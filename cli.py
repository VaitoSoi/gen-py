import argparse
import main
import yaml
import importlib.util
import re
import time
import os


def __import__(file_path: str):
    try:
        spec = importlib.util.spec_from_file_location("main", file_path)
        if spec is None:
            raise FileNotFoundError()
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        if hasattr(module, "main") and callable(module.main):
            return module.main
        else:
            raise AttributeError(
                "Function 'main' not found or not callable in the specified file."
            )
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def __main__():
    parser = argparse.ArgumentParser(
        description="Generate testcases for online judge", add_help=True, prog="cli"
    )
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument(
        "--generate",
        "-g",
        action="store_true",
        help="Generate testcases",
    )
    parser.add_argument(
        "--execute",
        "-e",
        action="store_true",
        help="Execute testcases",
    )
    parser.add_argument(
        "--zip",
        "-z",
        action="store_true",
        help="Zip testcases",
    )
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Save testcases",
    )
    parser.add_argument(
        "--count",
        "-c",
        help="Count testcases"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Not implemented yet :(",
    )

    args = parser.parse_args()
    with open(args.config, "r") as file:
        global config
        file = yaml.safe_load(file)
        test_count = file["TestCount"]
        test_code = file["TestCode"] or file["Code"]
        test_range = [
            main.Range(
                limit=ind["limit"] if isinstance(ind["limit"][0], int) else [tuple(i) for i in ind["limit"]],
                count=ind["count"],
                test_function=__import__(ind["test_function"] or ind["function"])
                if "test_function" in ind or "function" in ind
                else None,
                post_function=__import__(ind["post_function"])
                if "post_function" in ind or "post_function" in ind
                else None,
            )
            for ind in file["TestRange"]
        ]
        name = file["Name"]
        language = file["Language"]
        cpp_version = str(file["CppVersion"]) or "17"
        code_path = file["CodePath"]
        save_path = file["SavePath"]
        test_path = re.sub(r"\{(Name|name)\}", name, file["TestPath"] or "")
        oj_format_path = re.sub(r"\{(Name|name)\}", name, file["OJTestPath"]) or None
        test_zip_path = re.sub(r"\{(Name|name)\}", name, file["TestZipPath"]) or None
        oj_format_zip_path = (
            re.sub(r"\{(Name|name)\}", name, file["OJTestZipPath"]) or None
        )
        config = main.GenerateTestProperties(
            test_count=test_count,
            test_code=test_code,
            test_range=test_range,
            name=name,
            language=language,
            cpp_version=cpp_version,
            code_path=code_path,
            save_test_path=save_path,
            test_path=test_path,
            oj_format_path=oj_format_path,
            test_zip_path=test_zip_path,
            oj_format_zip_path=oj_format_zip_path,
        )

    generator = main.GenerateTest(config)

    if len(args.__dict__) == 1 or not any(args.__dict__.values()):  # No
        parser.print_help()
        return

    if args.generate:
        generator.generate_test()

    if args.execute:
        generator.execute()

    if args.zip:
        generator.zip()

    if args.save:
        generator.save()

    if args.count:
        count = 0
        for dir_name in os.listdir(config.test_path):
            dir_path = os.path.join(config.test_path, dir_name)
            if os.path.isdir(dir_path):
                with open(os.path.join(dir_path, f"{config.name}.OUT"), "r") as file:
                    if file.readline() == args.count:
                        count += 1
        print(f"Total testcase's outputs with content \"{args.count}\": {count}")

    if args.test:
        start = time.time()
        # dis.dis(generator.generate_test)
        generator.generate_test()
        end = time.time()
        print(f"Time taken: {end-start}s")


if __name__ == "__main__":
    __main__()