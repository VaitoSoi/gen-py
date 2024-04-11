from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple, Callable
import re
import random
import logging
import os
import sys
import shutil
import subprocess
import zipfile


class DefineVariables(BaseModel):
    name: str
    data_type: str  # "char" or "int"
    range: Tuple[str] | Tuple[float, float]


class Range(BaseModel):
    limit: List[float] | List[Tuple[float, float]]
    count: int | float
    test_function: Optional[Callable[[str, Any], bool]]
    post_function: Optional[Callable[[str, Any], bool]]


class GenerateTestProperties(BaseModel):
    test_count: int
    test_code: str
    test_range: List[Range]
    code_path: Optional[str] = None
    language: str
    cpp_version: str = "17"
    name: str
    save_test_path: Optional[str] = None
    test_path: str
    test_zip_path: Optional[str] = None
    oj_format_path: Optional[str] = None
    oj_format_zip_path: Optional[str] = None


class GenerateTest:
    # Định nghĩa:
    # + limit: giới hạn giá trị của biến
    # + range: giới hạn của từng testcase

    # Local variables
    cache: Dict[str, Any]
    define: Dict[str, DefineVariables]
    tmp: Dict[str, Any]
    log: logging.Logger = logging.getLogger("gen_py")

    # Properties
    test_count: int
    test_code: str
    test_range: List[Range]
    code_path: Optional[str]
    language: str
    cpp_version: str
    name: str
    save_test_path: Optional[str]
    test_path: str
    oj_format_path: Optional[str]
    test_zip_path: Optional[str]
    oj_format_zip_path: Optional[str]

    # Constants
    regex = {
        "split": r"-{5,}",
        "array_1d": r"\[(.+), (\d+|\w+)\]",
        "array_2d": r"\[(.+), (\d+|\w+) - (\d+|\w+)\]",
        "array_2d_split": r"\[(.+), (\d+|\w+) - (\d+|\w+), \"(.+|)\"\]",
        "array_sequence": r"\[(\d+|\w+) -> (\d+|\w+)\]",
        "array_reverse_sequence": r"\[(\d+|\w+) <- (\d+|\w+)\]",
        "range": r"\[(\d+|\w+) - (\d+|\w+)\]",
        "value": r"\[(\d+|\w+)\]",
        "eval": r"\{(.+)\}",
        "assign": r"const (\w+) = (.+)",
        "function": r"(.+)\((.+)\)",
    }

    def __init__(self, properties: GenerateTestProperties) -> None:
        self.test_count = properties.test_count
        self.test_code = properties.test_code
        self.test_range = properties.test_range
        self.code_path = properties.code_path
        self.language = properties.language
        self.cpp_version = properties.cpp_version
        self.name = properties.name
        self.save_test_path = properties.save_test_path
        self.test_path = properties.test_path
        self.test_zip_path = properties.test_zip_path
        self.oj_format_path = properties.oj_format_path
        self.oj_format_zip_path = properties.oj_format_zip_path

        if self.code_path is not None and not os.path.exists(self.code_path):
            raise FileNotFoundError(f"File not found: {self.code_path}")
        if self.language not in ["cpp", "python", "py"]:
            raise ValueError(f"Unsupport language: {self.language}")
        if self.language == "cpp" and self.cpp_version not in [
            "3",
            "11",
            "14",
            "17",
            "20",
        ]:
            raise ValueError(f"Unsupport cpp version: {self.cpp_version}")
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)
        if self.oj_format_path is not None and not os.path.exists(self.oj_format_path):
            os.makedirs(self.oj_format_path)

        self.cache = {}
        self.define = {}
        self.tmp = {}

        for ind in range(len(self.test_range)):
            count = self.test_range[ind].count
            if count <= 0:
                raise ValueError(f"Invalid count for test_range {ind + 1}")
            if count > 0 and count < 1:
                self.test_range[ind].count = int(count * self.test_count)

        self.parse_input_code(self.test_code)

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        stream_handler.setLevel(logging.INFO)
        self.log.setLevel(logging.INFO)
        self.log.addHandler(stream_handler)

    def parse_input_code(self, input: str) -> None:
        header, test_code = re.split(self.regex["split"], input)

        self.parse_header(header)
        self.test_code = test_code

    def parse_header(self, header: str) -> None:
        lines = header.splitlines()
        for line in lines:
            if line == "":
                continue
            args = line.split()
            name = args[0]
            data_type = args[1]
            range = (
                (args[2] if len(args) == 3 else args[2] + " ",)  # type: Tuple[str]
                if data_type == "char"
                else (float(args[2]), float(args[3]))  # type: Tuple[float, float]
                if data_type == "int"
                else None
            )

            if range is None:
                raise ValueError(f"Invalid data types for {name}")

            self.define[name] = DefineVariables(
                name=name, data_type=data_type, range=range
            )

    # Generating

    def generate_test(self, re_generate: List[int] = []) -> None:
        for root, dirs, files in os.walk(f"{os.getcwd()}/{self.test_path}"):
            for dir in dirs:
                ind = int(re.findall(r"\d+", dir)[0]) - 1
                if len(re_generate) == 0 or ind in re_generate:
                    shutil.rmtree(f"{root}/{dir}")
                    self.log.info(f"Cleaned test [TEST_{ind + 1}]...")
                ind += 1
        for root, dirs, files in os.walk(f"{os.getcwd()}/{self.oj_format_path}"):
            for file in files:
                if len(re_generate) == 0 or int(re.findall(r"\d+", file)[0]) - 1 in re_generate:
                    os.remove(f"{root}/{file}")
        self.log.info("Cleaning old testcases successfully!")

        self.log.info("Generating test...")
        configs: List[Range] = []
        if len(re_generate) == 0:
            total_count = 0
            for ind in self.test_range:
                tmp_last = total_count
                total_count += int(ind.count)
                for i in range(tmp_last, total_count):
                    configs.append(ind)

            self.tmp["configs"] = configs[: self.test_count]
        else:
            configs = self.tmp["configs"]

        if len(configs) < self.test_count:
            raise ValueError(f"missing range for test {len(configs) + 1}")

        code_lines = self.test_code.split("\n")


        def func(index: int, config: Range):
            self.log.info(f"[TEST {index + 1}] Generating...")
            test = ""
            while True:
                # self.tmp = {}
                self.tmp["range"] = config.limit
                test = ""
                for line in code_lines:
                    if line == "":
                        continue
                    test += self.parse_cmd_line(line) + "\n"
                if config.test_function is not None:
                    if config.test_function(test, index):
                        break
                    else:
                        self.log.info(f"[TEST {index + 1}] Regenerating...")
                else:
                    break
            test = "\n".join([ind for ind in test.split("\n") if ind.strip()])
            os.mkdir(f"{os.getcwd()}/{self.test_path}/TEST_{index + 1}")
            with open(
                f"{os.getcwd()}/{self.test_path}/TEST_{index + 1}/{self.name.upper()}.INP",
                "w",
            ) as f:
                f.write(test)
            if self.oj_format_path is not None:
                with open(
                    f"{os.getcwd()}/{self.oj_format_path}/{self.name.upper()}_{index + 1}.INP",
                    "w",
                ) as f:
                    f.write(test)
            
            return
    
        if len(re_generate) == 0:
            for index, config in enumerate(configs[: self.test_count]):
                func(index, config)
        else:
            for index in re_generate:
                func(index, configs[index])

        self.log.info("Writing testcases successfully!")

    # Parsing

    def parse_cmd_line(self, code: str, split: str = " ") -> str:
        # print(code)
        if (
            re.search(self.regex["array_1d"], code) is not None
            or re.search(self.regex["array_2d"], code) is not None
            or re.search(self.regex["array_2d_split"], code) is not None
        ):
            return self.parse_array(code)

        cmds = code.split(";")

        return split.join([self.parse_command(cmd) for cmd in cmds]) 
    
    def parse_array(self, code: str) -> str:
        result: str = ""

        args = (
            re.findall(self.regex["array_2d_split"], code)
            or re.findall(self.regex["array_2d"], code)
            or re.findall(self.regex["array_1d"], code)
        )[0]

        command: str = args[0]
        rows, columns = [
            int(ind) if ind not in self.cache else self.cache[ind]
            for ind in (
                [args[1], args[2]]
                if re.search(self.regex["array_2d_split"], code) is not None
                or re.search(self.regex["array_2d"], code) is not None
                else [args[1], 1]
            )
        ]
        split: str = (
            args[3]
            if re.search(self.regex["array_2d_split"], code) is not None
            else " "
        )
        if re.search(self.regex["array_sequence"], command) is not None:
            start, stop = [
                int(self.cache.get(ind, ind))
                for ind in re.findall(self.regex["array_sequence"], command)[0]
            ]
            if start < stop:
                raise ValueError("invalid sequence")
        elif re.search(self.regex["array_reverse_sequence"], command) is not None:
            start, stop = [
                int(self.cache.get(ind, ind))
                for ind in re.findall(self.regex["array_reverse_sequence"], command)[0]
            ]
            if start > stop:
                raise ValueError("invalid reverse sequence")

        last_value: str = ""
        for i in range(rows):
            value: str = ""
            if re.search(self.regex["array_sequence"], command) is not None:
                args = [
                    int(self.cache.get(ind, ind))
                    for ind in re.findall(self.regex["array_sequence"], command)[0]
                ]
                start = args[1]
                stop = int(last_value) if last_value != "" else args[0]
                value = (
                    str(
                        random.randint(
                            start, stop if last_value == "" else int(last_value)
                        )
                    )
                    if int(start) < int(stop) and int(start) < int(last_value or stop)
                    else str(start)
                )
                last_value = value
                value += split
            elif re.search(self.regex["array_reverse_sequence"], command) is not None:
                args = [
                    int(self.cache.get(ind, ind))
                    for ind in re.findall(
                        self.regex["array_reverse_sequence"], command
                    )[0]
                ]
                start = int(last_value) if last_value != "" else args[0]
                stop = args[1]
                value = (
                    str(random.randint(start, stop))
                    if int(start) < int(stop)
                    else str(start)
                )
                last_value = value
                value += split
            else:
                value = "".join([self.parse_cmd_line(command, split) for _ in range(columns)])

            result += f"{value}\n"

        return result

    def parse_command(self, code: str) -> str:
        if re.search(self.regex["assign"], code) is not None:
            key, cmd = re.findall(self.regex["assign"], code)[0]
            
            is_ghost = "ghost" in key
            if "ghost" in key:
                key = key.replace("ghost", "")
            key.strip()
            
            value = str(cmd if cmd.isdigit() else self.cache.get(cmd, self.parse_command(cmd)))
            self.cache[key] = value

            return value if not is_ghost else ""
        elif re.search(self.regex["value"], code) is not None:
            key = re.findall(self.regex["value"], code)[0][0]
            return str(self.cache.get(key, key))
        elif re.search(self.regex["range"], code) is not None:
            start, stop = [int(self.cache.get(ind, ind)) for ind in re.findall(self.regex["range"], code)[0]]
            return str(random.randint(min(start, stop), max(start, stop))) if start != stop else str(start)
        elif re.search(self.regex["eval"], code) is not None:

            def get(key: Any) -> Any:
                return self.cache.get(key, key)

            return str(eval(re.findall(self.regex["eval"], code)[0]))
        elif re.search(self.regex["function"], code) is not None:
            func, args = re.findall(self.regex["function"], code)[0]
            if func == "string":
                string, length = args.split(", ")
                return "".join(random.choices(self.define[string].range[0], k=int(self.cache.get(length, length)))) # type: ignore
            else:
                raise ValueError(f"Unsupport function: {func}")
        else:
            args = code.split()

            is_ghost = "ghost" in args
            if_cache = "const" in args

            [args.remove(ind) for ind in ["ghost", "const"] if ind in args]

            variable_define = self.define[args[0]]
            variable_limit = variable_define.range
            variable_range: List[float] = (
                self.tmp["range"]
                if not isinstance(self.tmp["range"][0], list) and not isinstance(self.tmp["range"][0], tuple)
                else self.tmp["range"][
                    list(self.define.keys()).index(variable_define.name)
                ]
            )
            # print(variable_define, variable_limit, variable_range)
            value = (
                random.choices(variable_limit[0]) # type: ignore
                if variable_define.data_type == "char"
                else random.randint(
                    int((variable_limit[1] - variable_limit[0]) * variable_range[0] + variable_limit[0]), # type: ignore
                    int((variable_limit[1] - variable_limit[0]) * variable_range[1] + variable_limit[0]), # type: ignore
                )
            )

            if if_cache:
                self.cache[args[0]] = value

            return str(value) if not is_ghost else ""

    # Utils

    def execute(self, white_list: List[int] = []) -> None:
        self.log.info("Compiling & executing code...")
        if not self.code_path:
            raise ValueError("code_path is not defined")

        executble_file = "main.exe" if os.name == "nt" else "main"
        executble_path = f"{"/".join(self.code_path.split("/")[:-1])}/{executble_file}"
        if self.language == "cpp":
            gpp_path = shutil.which("g++") or "g++"
            args = [
                gpp_path,
                f"{self.code_path}",
                "-std=c++" + self.cpp_version,
                "-Wall",
                "-march=native",
                "-O2",
                "-s",
                "-lm",
                "-o",
                executble_path,
            ]
            callback = subprocess.Popen(args, stderr=subprocess.PIPE)
            callback.wait()
            stderr = callback.communicate()
            code = callback.returncode

            if code != 0:
                self.log.error(f"Error: {stderr}")
                raise ValueError(f"g++ compiler throw code {code}")
            else:
                self.log.info("[COMPILE] Compiled successfully!")

        regenerates = []

        for root, dirs, files in os.walk(self.test_path):
            for dir in dirs:
                ind = int(re.findall(r"\d+", dir)[0]) - 1
                if len(white_list) > 0 and ind not in white_list:
                    continue
                if self.language == "py" or self.language == "python":
                    shutil.copy(f"{self.code_path}", f"{root}/{dir}/main.py")
                else:
                    shutil.copy(f"{executble_path}", f"{root}/{dir}/{executble_file}")

                python_path = shutil.which("python")
                callback = subprocess.Popen(
                    [python_path or "python", "main.py"]
                    if self.language == "py"
                    else [f"{executble_file}"],
                    shell=True,
                    cwd=f"{os.getcwd()}/{root}/{dir}",
                )
                callback.wait()

                stdout, stderr = callback.communicate()
                code = callback.returncode
                if code != 0:
                    self.log.error(stderr)
                    raise ValueError(f"executable file throw code {code} at test {ind + 1}")
                else:
                    self.log.info(f"[EXECUTE] Excuted test [{dir}] successfully!")

                if self.oj_format_path is not None:
                    shutil.copyfile(
                        f"{root}/{dir}/{self.name}.OUT",
                        f"{os.getcwd()}/{self.oj_format_path}/{self.name.upper()}_{ind + 1}.OUT",
                    )

                if ("configs" in self.tmp) and self.tmp["configs"][ind].post_function is not None:
                    with open(f"{root}/{dir}/{self.name}.OUT", "r") as f:
                        stdout = f.read()
                        cb = self.tmp["configs"][ind].post_function(stdout)
                        if not cb:
                            regenerates.append(ind)

        if len(regenerates) > 0:
            self.log.info("Regenerating testcases due to post_function request...")
            self.log.info(f"Regenrate testcases: {", ".join([str(ind) for ind in regenerates])}\n")
            self.generate_test(regenerates)
            self.execute(regenerates)

    def zip(self) -> None:
        if self.test_zip_path is None:
            raise ValueError("test_zip_path is not defined")

        self.log.info("Zipping testcases...")
        with zipfile.ZipFile(self.test_zip_path, "w") as zipf:
            for root, dirs, files in os.walk(self.test_path):
                for dir in dirs:
                    for sub_root, sub_dirs, sub_files in os.walk(f"{root}/{dir}"):
                        for file in sub_files:
                            zipf.write(f"{sub_root}/{file}", f"{dir}/{file}")
            zipf.close()
        self.log.info("Zipped testcases successfully!")

        if self.oj_format_path is not None and self.oj_format_zip_path is not None:
            self.log.info("Zipping oj format...")
            with zipfile.ZipFile(self.oj_format_zip_path, "w") as zipf:
                for root, dirs, files in os.walk(self.oj_format_path):
                    for file in files:
                        zipf.write(f"{root}/{file}", file)
                zipf.close()
            self.log.info("Zipped oj format successfully!")

    def save(self):
        if self.save_test_path is None:
            raise ValueError("save_test_path is not define")

        if self.test_zip_path is not None:
            shutil.copyfile(
                f"{os.getcwd()}/{self.test_zip_path}",
                f"{os.getcwd()}/{self.save_test_path}/{self.name}.zip",
            )

        if self.oj_format_zip_path is not None:
            shutil.copyfile(
                f"{os.getcwd()}/{self.oj_format_zip_path}",
                f"{os.getcwd()}/{self.save_test_path}/{self.name}_OJ.zip",
            )
