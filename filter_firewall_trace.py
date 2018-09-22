import datetime
import re
from typing import List


class TableChunk:

    def __init__(self, table_chunk: str) -> None:
        table_chunk = table_chunk.split(':')
        self.table = table_chunk[0]
        self.chain = table_chunk[1]
        self.nature = table_chunk[2]
        self.rule_num = int(table_chunk[3])
        pass

    def __str__(self) -> str:
        return '%s:%s:%s:%s' % (self.table, self.chain, self.nature, self.rule_num)


class LogItem:

    def __init__(self, line: str) -> None:
        self.components: List[str] = line.split(' ')

        time = self.components[2].split(':')
        hour = int(time[0])
        minute = int(time[1])
        sec = int(time[2])
        month = self.components[0]
        if month == "Sep":
            month = 9
        else:
            raise Exception('Support Sep only for now')
        day = int(self.components[1])
        self.time = datetime.datetime(2018, month, day, hour, minute, sec)

        # print('%s | %s' % (self.components[8], self.components[9]))
        table_chunk = re.search('TRACE: (.*?)\sIN=', line)
        table_chunk = table_chunk.group(1) if table_chunk else None
        self.table_chunk = TableChunk(table_chunk)

        in_ = re.search('IN=(.*?)\s', line)
        self.in_ = in_.group(1) if in_ else None
        # self.in_ = self.components[9].split('=')[1]

        out_ = re.search('OUT=(.*?)\s', line)
        self.out_ = out_.group(1) if out_ else None
        # self.out_ = self.components[10].split('=')[1]

        src = re.search('SRC=(.*?)\s', line)
        self.src = src.group(1) if src else None
        # self.src = self.components[12].split('=')[1]

        dst = re.search('DST=(.*?)\s', line)
        self.dst = dst.group(1) if dst else None
        # self.dst = self.components[13].split('=')[1]

        id_ = re.search('ID=(.*?)\s', line)
        self.id = int(id_.group(1)) if id_ else None

        proto = re.search('PROTO=(.*?)\s', line)
        self.proto = proto.group(1) if proto else None
        # self.proto = self.components[20].split('=')[1]

        dpt = re.search('DPT=(.*?)\s', line)
        self.dpt = int(dpt.group(1)) if dpt else None
        # self.dpt = self.components[22].split('=')[1]

    def __str__(self) -> str:
        return '%s %s IN=%s, OUT=%s SRC=%s DST=%s ID=%s PROTO=%s DPT=%s' % \
               (self.time, self.table_chunk, self.in_, self.out_, self.src, self.dst, self.id, self.proto, self.dpt)


class GetSpecificLine:

    def __init__(self, file_name: str = '/var/log/kern.log', desired_time: str = None,
                 desired_table: str = 'filter', desired_chain: str = None, desired_src: str = None,
                 desired_id: int = None, desired_dpt: int = 80) -> None:
        if desired_time:
            desired_time = datetime.datetime.strptime(desired_time, '%H:%M:%S')
        with open(file_name) as f:
            # i = 0
            for line in f:
                # print(i)
                if line.find('TRACE:') == -1:
                    continue
                log_item = LogItem(line)

                if desired_time:
                    desired_time_l = desired_time
                else:
                    desired_time_l = log_item.time

                if desired_chain:
                    desired_chain_l = desired_chain
                else:
                    desired_chain_l = log_item.table_chunk.chain

                desired_table_l = desired_table

                if desired_src:
                    desired_src_l = desired_src
                else:
                    desired_src_l = log_item.src

                if desired_id:
                    desired_id_l = desired_id
                else:
                    desired_id_l = log_item.id

                desired_dpt_l = desired_dpt

                if log_item.time == desired_time_l and \
                        log_item.table_chunk.table == desired_table_l and \
                        log_item.src == desired_src_l and \
                        log_item.table_chunk.chain == desired_chain_l and \
                        log_item.id == desired_id_l and \
                        log_item.dpt == desired_dpt_l:
                    print(log_item)
                # i = i + 1


if __name__ == '__main__':
    GetSpecificLine(desired_table='filter', desired_src='113.161.93.107', desired_id=21808, )
