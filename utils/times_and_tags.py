import time
from tabulate import tabulate


class times_and_tags:
    def __init__(self):
        # keeps track of absolute times
        self.times = []
        # keeps track of all tags
        self.tags = []
        # keeps track of number of calls of the differnet tags
        self.num_calls = []
        # keeps track of relative times of the different tags
        self.tag_times = []
        # keeps track of relative times of the different tags (for the first call)
        self.tag_times_first_call = []
        # keep track of old tag
        self.old_tag = None

        # just a flag to track self.end_add
        self.add_ended = False

    # def add(self, tag):
    #     self.times.append(time.time())
    #     self.tags.append(tag)

    def add(self, tag: str):
        # Skip this for the very fist call (old tag == None)
        if self.old_tag is not None:
            if self.old_tag in self.tags:
                idx = self.tags.index(self.old_tag)
                # counts the number of calls of specific tag
                self.num_calls[idx] += 1
                # calc (and add) delta t
                now = time.time()
                self.tag_times[idx] += now - self.times[-1]
                # store absolute time
                self.times.append(now)
            else:
                # counts the number of calls of specific tag
                self.num_calls.append(1)
                # create new tag
                self.tags.append(self.old_tag)
                # calc (and create new entry for) delta t
                now = time.time()
                delta_t = now - self.times[-1]
                self.tag_times.append(delta_t)
                # save for the first call
                self.tag_times_first_call.append(delta_t)
                # store absolute time
                self.times.append(now)
        else:
            self.times.append(time.time())

        # save tag for next call
        self.old_tag = tag

    def end_add(self):
        # call this function to end the time tracking, but you want to print the results later
        # i.e. print of table != end time of last tracking
        self.add("")
        self.add_ended = True

    def print(self):
        # add last time (if needed)
        if self.add_ended:
            pass
        else:
            self.end_add()

        # if no repeated counting
        if all(1 == x for x in self.num_calls):
            self._print_simple()
        else:
            self._print_repeated()

    def _print_simple(self):
        # delta t
        dt = self.tag_times

        # delta t percentage
        sum_dt = sum(dt)
        p_dt = [x * 100 / sum_dt for x in dt]

        # create data for table
        data = []
        headers = ["Pos", "Tag", "Time [s]", "Time [%]"]
        for i in range(len(self.times) - 1):
            data.append([i, self.tags[i], dt[i], p_dt[i]])
        data.append(["x", "SUM", sum_dt, 100])

        # print table
        print("")
        print(tabulate(data, headers=headers, tablefmt='orgtbl'))
        print("")

    def _print_repeated(self):
        # delta t
        dt = [i / j for i, j in zip(self.tag_times, self.num_calls)]

        # delta t percentage (on average)
        sum_dt = sum(dt)
        p_dt = [x * 100 / sum_dt for x in dt]

        # sum num calls
        sum_calls = sum(self.num_calls)

        # sum time first call
        sum_first_call = sum(self.tag_times_first_call)

        # create data for table
        data = []
        headers = ["Pos", "Tag", "#Calls", "Time 1.c. [s]", "Time p.c. [s]", "Time [%]"]
        for i in range(len(self.tags)):
            data.append([i, self.tags[i], self.num_calls[i], self.tag_times_first_call[i], dt[i], p_dt[i]])
        data.append(["x", "SUM", sum_calls, sum_first_call, sum_dt, 100])

        # print table
        print("")
        print(tabulate(data, headers=headers, tablefmt='orgtbl'))
        print("")

    def reset(self):
        self.times = []
        self.tags = []
