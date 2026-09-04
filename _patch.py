ROOT = r"c:\Users\user\Downloads\SIH"
p = ROOT + r"\backend\kfetch.py"
src = open(p, encoding="utf-8").read()
old = '''            if not download_dataset_file(owner, dslug, cand, tmp):
                last_err = "HTTP failure"
                continue
            if _extract_member(tmp, os.path.basename(cand), out_path):
                os.remove(tmp)
                return True
            last_err = "member not in file"'''
new = '''            with open(tmp, "rb") as fh:
                magic = fh.read(2)
            if magic != b"PK":
                os.replace(tmp, out_path)      # per-file download IS the member
                return True
            if _extract_member(tmp, os.path.basename(cand), out_path):
                os.remove(tmp)
                return True
            last_err = "member not in file"'''
assert old in src, "anchor missing"
open(p, "w", encoding="utf-8", newline="").write(src.replace(old, new))
print("kfetch fixed")
