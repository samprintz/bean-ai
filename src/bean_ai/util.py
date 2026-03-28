def confirm_overwrite(path: str) -> bool:
    answer = input(f"{path} already exists. Overwrite? [y/N] ").strip().lower()
    return answer == "y"
