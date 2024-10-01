import uuid

class linkedString:
    def __init__(self, string):
        prev_address = "START"
        self.string_map = {"START": {"value": string[0]}}

        if len(string) > 2:
            for s in string[1:-1]:
                address = str(uuid.uuid4())
                self.string_map[prev_address]["next"] = address
                self.string_map[address] = {"value": s, "previous": prev_address}
                prev_address = address

        if len(string) > 1:
            self.string_map[prev_address]["next"] = "END"
            self.string_map["END"] = {"value": string[-1], "previous": prev_address, "next": "START"}
            self.string_map["START"]["previous"] = "END"

        else:
            self.string_map["START"]["next"] = "START"
            self.string_map["START"]["previous"] = "START"

    def index(self, idx):
        return linkedIndex(self, idx)

    def __getitem__(self, idx):
        if type(idx) == int:
            return self.__getitem__(self.index(idx))
        if type(idx) == slice:
            if type(idx.start) == int or type(idx.stop) == int:
                if type(idx.start) == int:
                    start = self.index(idx.start)
                elif idx.start is None:
                    start = self.index(0)
                else:
                    start = idx.start
                if type(idx.stop) == int:
                    stop = self.index(idx.stop)
                elif idx.stop is None:
                    stop = self.index(-1)
                else:
                    stop = idx.stop
                return self.__getitem__(slice(start, stop))

            elif type(idx.start) == linkedIndex and type(idx.stop) == linkedIndex:
                ret = ""
                current = idx.start
                while current != idx.stop:
                    ret += self[current]
                    current = current + 1
                return ret

        if type(idx) == linkedIndex:
            return idx.linked_string.string_map[idx.current_address]["value"]

    def __setitem__(self, idx, value):
        if type(value) != str or len(value) != 1:
            raise ValueError("linkedString can only be assigned a single character")
        if type(idx) == int:
            return self.__setitem__(self.index(idx), value)
        if type(idx) == slice:
            if type(idx.start) == int or type(idx.stop) == int:
                if type(idx.start) == int:
                    start = self.index(idx.start)
                elif idx.start is None:
                    start = self.index(0)
                else:
                    start = idx.start
                if type(idx.stop) == int:
                    stop = self.index(idx.stop)
                elif idx.stop is None:
                    stop = self.index(-1)
                else:
                    stop = idx.stop
                return self.__setitem__(slice(start, stop), value)

            elif type(idx.start) == linkedIndex and type(idx.stop) == linkedIndex:
                current = idx.start
                while current != idx.stop:
                    self[current] = value
                    current = current + 1

        if type(idx) == linkedIndex:
            self.string_map[idx.current_address]["value"] = value

    def replace(self, from_idx, to_idx, value):
        if type(from_idx) != linkedIndex or type(to_idx) != linkedIndex:
            raise TypeError("from_idx and to_idx must be linkedIndex objects")
        current = from_idx
        while current != to_idx:
            self[current] = value
            current = current + 1

    def insert(self, idx, value):
        if type(idx) != linkedIndex:
            raise TypeError("idx must be a linkedIndex object")
        new_address = str(uuid.uuid4())
        self.string_map[new_address] = {"value": value, "previous": idx.current_address, "next": self.string_map[idx.current_address]["next"]}
        self.string_map[self.string_map[idx.current_address]["next"]]["previous"] = new_address
        self.string_map[idx.current_address]["next"] = new_address
        idx.current_address = new_address

    def delete(self, idx_object):
        if type(idx_object) != linkedIndex:
            raise TypeError("idx must be a linkedIndex object")
        self.string_map[self.string_map[idx_object.current_address]["previous"]]["next"] = self.string_map[idx_object.current_address]["next"]
        self.string_map[self.string_map[idx_object.current_address]["next"]]["previous"] = self.string_map[idx_object.current_address]["previous"]
        old_address = idx_object.current_address
        idx_object.current_address = self.string_map[idx_object.current_address]["previous"]
        del self.string_map[old_address]

    def __iter__(self):
        current = self.index(0)
        end = self.index(-1)
        while current != end:
            yield self[current]
            current = current + 1
        yield self[end]

    def __str__(self):
        return "".join([s for s in self])


# To implement: addition; what happens when index's portion of string is deleted
class linkedIndex:
    def __init__(self, linked_string, index):
        self.linked_string = linked_string
        if type(index) == int:
            if index >= 0:
                self.current_address = "START"
                for _ in range(index):
                    self.current_address = self.linked_string.string_map[self.current_address]["next"]
            else:
                self.current_address = "END"
                for _ in range(abs(index)-1):
                    self.current_address = self.linked_string.string_map[self.current_address]["previous"]
        elif type(index) == str:
            self.current_address = index

    def __add__(self, other):
        if type(other) == int:
            if other >= 0:
                new_index = self.current_address
                for _ in range(other):
                    try:
                        new_index = self.linked_string.string_map[new_index]["next"]
                    except KeyError:
                        return linkedIndex(self.linked_string, "END")
                return linkedIndex(self.linked_string, new_index)
            else:
                return self.__sub__(abs(other))
        else:
            raise TypeError("unsupported operand type(s) for +: 'linkedIndex' and '{}'".format(type(other).__name__))

    def __sub__(self, other):
        if type(other) == int:
            if other >= 0:
                new_index = self.current_address
                for _ in range(other):
                    try:
                        new_index = self.linked_string.string_map[new_index]["previous"]
                    except KeyError:
                        return linkedIndex(self.linked_string, "START")
                return linkedIndex(self.linked_string, new_index)
            else:
                return self.__add__(abs(other))
        else:
            raise TypeError("unsupported operand type(s) for -: 'linkedIndex' and '{}'".format(type(other).__name__))

    def __eq__(self, other):
        if type(other) == linkedIndex:
            return self.current_address == other.current_address
        return False

if __name__ == "__main__":
    s = linkedString("Hello, World!")
    idx = s.index(5)
    for i in range(3):
        s.insert(idx, "!")
    print(s)
    s.delete(idx)
    print(s)
