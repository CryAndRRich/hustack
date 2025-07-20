from typing import List

class SQL:
    def __init__(self, names: List[str], columns: List[int]):
        self.tables = {}
        for name, col in zip(names, columns):
            self.tables[name] = {
                "cols": col,
                "rows": {},
                "next_id": 1
            }

    def ins(self, name: str, row: List[str]) -> bool:
        if name not in self.tables or len(row) != self.tables[name]["cols"]:
            return False
        t = self.tables[name]
        t["rows"][t["next_id"]] = row
        t["next_id"] += 1
        return True

    def rmv(self, name: str, rowId: int) -> None:
        if name in self.tables:
            self.tables[name]["rows"].pop(rowId, None)

    def sel(self, name: str, rowId: int, columnId: int) -> str:
        if name not in self.tables:
            return "<null>"
        row = self.tables[name]["rows"].get(rowId)
        if not row or columnId < 1 or columnId > len(row):
            return "<null>"
        return row[columnId - 1]

    def exp(self, name: str) -> List[str]:
        if name not in self.tables:
            return []
        result = []
        for rowId in sorted(self.tables[name]["rows"]):
            row = self.tables[name]["rows"][rowId]
            result.append(str(rowId) + "," + ",".join(row))
        return result

# Your SQL object will be instantiated and called as such:
# obj = SQL(names, columns)
# param_1 = obj.ins(name,row)
# obj.rmv(name,rowId)
# param_3 = obj.sel(name,rowId,columnId)
# param_4 = obj.exp(name)