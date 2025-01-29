import pandas as pd


class Tributary:

    def __init__(self,name):
        self.reach_list = [] # list of reach id
        self.name = name

    def add_reach(self,id):
        self.reach_list.append(id)


class SWATReachReader:

    def __init__(self,df,mode):
        """
        :param df: SWAT river attribute table
        :param mode: "ASCENDING" OR "DESCENDING" depends on the sequence of the grid code
        ASCENDING (Grid Code):  Upstream -> 1 -> 2 -> 3 -> ... -> Downstream
        DESCENDING (Grid Code): Upstream -> 10 -> 9 -> 8 -> ... -> Downstream
        """
        self.mode = mode
        self.metadata = df
        self.objid = df["OBJECTID"]
        self.fromnode = df["FROM_NODE"]
        self.tonode = df["TO_NODE"]
        self.upper_nodes, self.upper_reach = self.find_upper()
        self.tributaries = []
        self.find_tributary()


    def find_upper(self):
        # check the TO_NODE to identify the upper nodes
        upper_nodes = []
        upper_reach = []
        for i in self.objid:
            if len(self.tonode[self.tonode == i]) == 0: # independent tributary:
                upper_nodes.append(i)
        for u in upper_nodes:
            reach_id = int(self.objid[self.fromnode == u].values)
            upper_reach.append(reach_id)
        return upper_nodes, upper_reach

    def find_tributary(self):
        used_reach = []
        def find_next(lreach,triobj,metadata,rid):
            reachinfo = metadata[metadata["OBJECTID"] == rid]
            t = int(reachinfo["TO_NODE"].values)
            next_reach_info = metadata[metadata["FROM_NODE"] == t]
            if len(next_reach_info) > 0:
                next_reach = int(next_reach_info["OBJECTID"].values)
                if next_reach not in lreach:
                    triobj.add_reach(next_reach)
                    lreach.append(next_reach)
                    find_next(lreach,triobj,metadata,next_reach)
        if self.mode == "ASCENDING":
            sequence = self.upper_reach
        elif self.mode == "DESCENDING":
            sequence = reversed(self.upper_reach)
        else:
            raise NotImplementedError("Only accept 'ASCENDING' or 'DESCENDING' mode")
        for r in sequence:
            # search from the highest node to make sure that the main channel is the longest
            used_reach.append(r)
            tri = Tributary(name="T{}".format(r))
            tri.add_reach(int(r))
            find_next(used_reach,tri,self.metadata,r)
            self.tributaries.append(tri)

    def write_tributaries(self,path):
        with open(path,"w") as f:
            for t in self.tributaries:
                if len(t.reach_list) > 1:
                    for i in range(len(t.reach_list)):
                        t.reach_list[i] = str(t.reach_list[i])
                    f.write(t.name + ': ' + ",".join(t.reach_list) + "\n")
                else:
                    f.write(t.name + ': ' + str(t.reach_list[0]) + "\n")

    def write_flow_function(self,path):
        # xlsx
        frame_data = []
        for t in self.tributaries:
            row = {"Reach":f"Reach{t.reach_list[0]}","Function":t.name,"Interpolation":0,"Scale Factor":1.0000,"Bound":0}
            frame_data.append(row)

        for n in self.objid:
            if int(n) not in self.upper_reach:
                row = {"Reach":f"Reach{int(n)}","Function": "S{}".format(int(n)), "Interpolation": 0, "Scale Factor": 1.0000, "Bound": 0}
                frame_data.append(row)
        df = pd.DataFrame(frame_data)
        df.to_excel(path)

    def write_segment_pairs(self,path):
        # xlsx
        with pd.ExcelWriter(path,engine='xlsxwriter') as writer:
            for r in self.tributaries:
                dfdata = []
                for id,t in enumerate(r.reach_list):
                    if id == 0:
                        row = {"From":0,"To":t,"Fraction":1}
                    elif id < len(r.reach_list) - 1:
                        row = {"From":t,"To":r.reach_list[id+1],"Fraction":1}
                    else:
                        reachinfo = self.metadata[self.objid == int(t)]
                        if int(reachinfo["TO_NODE"].values) == 0:
                            row = {"From": t, "To": 0, "Fraction": 1}
                        else:
                            tar = int(self.metadata[self.fromnode == int(reachinfo["TO_NODE"].values)]["OBJECTID"].values)
                            row = {"From": t, "To": tar, "Fraction": 1}
                    dfdata.append(row)
                    if id == 0 and len(r.reach_list) > 1:
                        rowa = {"From":t,"To":r.reach_list[id+1],"Fraction":1}
                        dfdata.append(rowa)
                if len(r.reach_list) == 1:
                    reachinfo = self.metadata[self.objid == t]
                    if int(reachinfo["TO_NODE"].values) == 0:
                        row = {"From": t, "To": 0, "Fraction": 1}
                    else:
                        tar = int(self.metadata[self.fromnode == int(reachinfo["TO_NODE"].values)]["OBJECTID"].values)
                        row = {"From": t, "To": tar, "Fraction": 1}
                    dfdata.append(row)
                df = pd.DataFrame(dfdata).astype(int)
                df.to_excel(writer,sheet_name=r.name)



if "__name__" == "__main__":
    df = pd.read_excel(r"D:\GrandWasp\river.xlsx") # The attribute table of the river shape file generated by ArcSWAT or QSWAT.
    reader = SWATReachReader(df,mode="ASCENDING") # The sequence of the grid code of the river channels in SWAT, can be "ASCENDING" or "DESCENDING" order.
    reader.write_tributaries(r"D:\GrandWasp\tributaries.txt") # The identified tributary information for your reference.
    reader.write_flow_function(r"D:\GrandWasp\FlowFunc.xlsx") # The definition of flow function, which can be directly copied and pasted in WASP8.
    reader.write_segment_pairs(r"D:\GrandWasp\GrandSegmentPairs.xlsx") # The identified segment pairs (river sequence) for WASP8.
