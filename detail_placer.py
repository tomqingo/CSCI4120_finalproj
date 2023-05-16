import os
import math
import argparse
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

class Cell:
    def __init__(self, locX, locY, width, height):
        assert isinstance(locX, float) or isinstance(locX, int)
        assert isinstance(locY, float) or isinstance(locY, int)
        assert isinstance(width, int) and isinstance(height, int)
        assert height in [8, 12]

        self.locX = locX
        self.locY = locY
        self.width = width
        self.height = height

        self.prevCell = -1
        self.laterCell = -1

class DetailPlacer:
    def __init__(self, nRows, nTracks, nSites, nCells, vCells, name2CellId, trackHeight, siteWidth):
        self.nRows = nRows
        self.nTracks = nTracks
        self.nSites = nSites
        self.nCells = nCells
        self.vCells = vCells
        self.name2CellId = name2CellId
        self.trackHeight = trackHeight
        self.siteWidth = siteWidth
    
    def takeSecond(self,elem):
        return elem[1]
    
    def PriorityDetermine(self):
        total_area_8T = 0
        total_area_12T = 0
        for id, cell in enumerate(self.vCells):
            if cell.height == 8:
                total_area_8T += cell.height*cell.width
            else:
                total_area_12T += cell.height*cell.width
        
        if total_area_8T>total_area_12T:
            return 1
        else:
            return 0
    
    def RowHeightAssignment(self):
        rowLabel = []
        distancemetric = []
        
        total_8T_width = 0
        total_12T_width = 0
        for id, cell in enumerate(self.vCells):
            if cell.height == 8:
                total_8T_width += cell.width
            else:
                total_12T_width += cell.width
        
        min_num_8T = math.ceil(total_8T_width*1.0 / self.nSites)*2
        min_num_12T = math.ceil(total_12T_width*1.0 / self.nSites)*3
        
        epsilon = 1.0

        #ratio = 2*1.0/3  #Suppose the left row would be assigned with 2/3 for 12T and 1/3 for 8T cells
        ratio = total_8T_width*8*1.0/(total_8T_width*8+total_12T_width*12)
        row_left = self.nRows - min_num_8T - min_num_12T
        num_row_8T = int(min_num_8T + int(row_left*ratio))
        num_row_12T = self.nRows - num_row_8T

        #print(min_num_8T)
        #print(min_num_12T)
        #print(num_row_8T)
        #print(num_row_12T)
        #print(self.nRows)

        first_order_id = self.PriorityDetermine()

        for id in range(self.nRows):
            if first_order_id == 1:
                rowLabel.append(0)
            else:
                rowLabel.append(1)

        for id in range(self.nRows):
            total_dis = 0
            for cellid in range(self.nCells):
                if abs(self.vCells[cellid].locY-id*4) < 1.0*epsilon:
                    if first_order_id == 1:
                        if self.vCells[cellid].height == 12:
                            total_dis += self.vCells[cellid].height*self.vCells[cellid].width
                        else:
                            total_dis -= self.vCells[cellid].height*self.vCells[cellid].width
                    else:
                        if self.vCells[cellid].height == 8:
                            total_dis += self.vCells[cellid].height*self.vCells[cellid].width
                        else:
                            total_dis -= self.vCells[cellid].height*self.vCells[cellid].width                        
            distancemetric.append((id,total_dis))
        
        distancemetric.sort(key=self.takeSecond, reverse=True)

        if first_order_id == 1:
            for id in range(num_row_12T):
                rowLabel[distancemetric[id][0]] = 1
        else:
            for id in range(num_row_8T):
                rowLabel[distancemetric[id][0]] = 0            
        
        print(rowLabel)

        #KMeans to refine the rowLabel

        #Some refinements about the rows
        row_label_order = []
        row_label_num = []

        diff_Label = []

        for id in range(len(rowLabel)-1):
            diff_Label.append(rowLabel[id+1]-rowLabel[id])

        cnt  = 0
        for id in range(len(diff_Label)):
            if diff_Label[id]!=0:
                row_label_order.append(rowLabel[id])
                row_label_num.append(cnt+1)
                cnt = 0
            else:
                cnt = cnt + 1
        
        row_label_order.append(rowLabel[-1])
        if cnt != 0:
            row_label_num.append(cnt+1)
        else:
            row_label_num.append(1)
        #print(row_label_num)
        #print(row_label_order)
        for id in range(len(row_label_order)):
            if row_label_order[id] == 0:
                row_label_num[id] = int(math.ceil(row_label_num[id]*1.0/2)*2)
            else:
                row_label_num[id] = int(math.ceil(row_label_num[id]*1.0/3)*3)
        
        total = 0
        total_12T = 0
        total_8T = 0

        for id in range(len(row_label_num)):
            total = total + row_label_num[id]
            if row_label_order[id] == 1:
                total_12T = total_12T + row_label_num[id]
            else:
                total_8T = total_8T + row_label_num[id]
        #print(row_label_order)
        #print(row_label_num)
        #print(total_12T, total_8T)
        #print(min_num_12T, min_num_8T)
        #print(self.nRows)
        # Reduce the most non-critical label
        while total > self.nRows:
            if (total_12T-min_num_12T)*1.0/3 < (total_8T-min_num_8T)*1.0/2:
                flag = 0
            else:
                flag = 1
            candidate_value = []
            candidate_id = []
            for subid in range(len(row_label_order)):
                if row_label_order[subid] == flag:
                    candidate_value.append(row_label_num[subid])
                    candidate_id.append(subid)
            max_id = candidate_id[candidate_value.index(max(candidate_value))]
            #max_id = row_label_num.index(max(row_label_num))
            if row_label_order[max_id] == 0:
                row_label_num[max_id] = row_label_num[max_id] - 2
                total = total - 2
                total_8T = total_8T - 2
            else:
                row_label_num[max_id] = row_label_num[max_id] - 3
                total = total - 3
                total_12T = total_12T - 3
        
        #print(total_12T, total_8T)
        #print(row_label_num)

        rowplacepermit = []
        rowLabelnew = []
        for id in range(len(row_label_num)):
            for subid in range(row_label_num[id]):
                if row_label_order[id] == 1:
                    rowLabelnew.append(12)
                else:
                    rowLabelnew.append(8)
                if row_label_order[id] == 1:
                    div = 3
                else:
                    div = 2
                if subid % div == 0:
                    rowplacepermit.append(1)
                else:
                    rowplacepermit.append(0)
        
        self.rowLabel = rowLabelnew
        self.rowplacepermit = rowplacepermit

        #for id in range(self.nRows):
        #    if first_order_id == 1:
        #        rowLabel[id] = 0
        #    else:
        #        rowLabel[id] = 1

        #if first_order_id == 1:
        #    for id in range(total_12T):
        #        rowLabel[distancemetric[id][0]] = 1
        #else:
        #    for id in range(total_8T):
        #        rowLabel[distancemetric[id][0]] = 0 

        #cnt_wrong = 0
        #cnt_total = 0
        #for id in range(len(self.rowplacepermit)):
        #    if self.rowplacepermit[id] == 1:
        #        cnt_total = cnt_total + 1
        #        if rowLabel[id] == 1:
        #            tmp = 12
        #        else:
        #            tmp = 8
        #        if self.rowLabel[id] != tmp:
        #            cnt_wrong = cnt_wrong + 1
        
        #print(cnt_total)
        #print(cnt_wrong)
        print(self.rowLabel)
        #print(self.rowplacepermit)

    def KMeansRowHeightAssignment(self):
        rowLabel = []
        distancemetric = []
        
        total_8T_width = 0
        total_12T_width = 0
        for id, cell in enumerate(self.vCells):
            if cell.height == 8:
                total_8T_width += cell.width
            else:
                total_12T_width += cell.width
        
        min_num_8T_tracks = math.ceil(total_8T_width*1.0 / self.nSites)*8
        min_num_12T_tracks = math.ceil(total_12T_width*1.0 / self.nSites)*12
        
        epsilon = 1.0

        #ratio = 2*1.0/3  #Suppose the left row would be assigned with 2/3 for 12T and 1/3 for 8T cells
        ratio = total_8T_width*8*1.0/(total_8T_width*8+total_12T_width*12)
        row_left = self.nTracks - min_num_8T_tracks - min_num_12T_tracks
        num_row_8T_tracks = int(min_num_8T_tracks + int(row_left*ratio))
        num_row_12T_tracks = self.nRows - num_row_8T_tracks

        #print(min_num_8T)
        #print(min_num_12T)
        #print(num_row_8T)
        #print(num_row_12T)
        #print(self.nRows)

        first_order_id = self.PriorityDetermine()

        for id in range(self.nTracks):
            if first_order_id == 1:
                rowLabel.append(0)
            else:
                rowLabel.append(1)

        #KMeans to refine the rowLabel
        Ycorr_min = []
        for id in range(len(self.vCells)):
            if first_order_id == 1:
                if self.vCells[id].height == 12:
                    Ycorr_min.append([self.vCells[id].locY])
            else:
                if self.vCells[id].height == 8:
                    Ycorr_min.append([self.vCells[id].locY])

        Ycorr_arr = np.array(Ycorr_min)
        #print(Ycorr_arr)

        if first_order_id == 1:
            kmeans = KMeans(n_clusters=num_row_12T_tracks).fit(Ycorr_arr)
        else:
            kmeans = KMeans(n_clusters=num_row_8T_tracks).fit(Ycorr_arr)

        cluster_centers = kmeans.cluster_centers_
        for id in range(len(cluster_centers)):
            row_id = int(math.ceil(cluster_centers[id][0]))
            start = row_id
            if start<self.nRows and rowLabel[start] == first_order_id:
                rowLabel[start] = 1-first_order_id
                iter_val = start
                while iter_val<self.nTracks and rowLabel[iter_val] == first_order_id:
                      iter_val = iter_val + 1
                if iter_val<self.nTracks:
                    rowLabel[iter_val] = first_order_id
                iter_val = start
                while iter_val>=0 and rowLabel[iter_val] == first_order_id:
                      iter_val = iter_val-1
                if iter_val>=0:
                    rowLabel[iter_val] = first_order_id                
            else:
                rowLabel[start] = first_order_id
        
        #Some refinements about the rows
        row_label_order = []
        row_label_num = []

        diff_Label = []

        for id in range(len(rowLabel)-1):
            diff_Label.append(rowLabel[id+1]-rowLabel[id])

        cnt  = 0
        for id in range(len(diff_Label)):
            if diff_Label[id]!=0:
                row_label_order.append(rowLabel[id])
                row_label_num.append(cnt+1)
                cnt = 0
            else:
                cnt = cnt + 1
        
        row_label_order.append(rowLabel[-1])
        if cnt != 0:
            row_label_num.append(cnt+1)
        else:
            row_label_num.append(1)
        #print(row_label_num)
        #print(row_label_order)
        for id in range(len(row_label_order)):
            if row_label_order[id] == 0:
                row_label_num[id] = int(math.ceil(row_label_num[id]*1.0/8)*8)
            else:
                row_label_num[id] = int(math.ceil(row_label_num[id]*1.0/12)*12)
        
        total = 0
        total_12T = 0
        total_8T = 0

        for id in range(len(row_label_num)):
            total = total + row_label_num[id]
            if row_label_order[id] == 1:
                total_12T = total_12T + row_label_num[id]
            else:
                total_8T = total_8T + row_label_num[id]
        #print(row_label_order)
        #print(row_label_num)
        print(total_12T, total_8T)
        print(min_num_12T_tracks, min_num_8T_tracks)
        print(self.nTracks)
        #print(self.nRows)
        # Reduce the most non-critical label
        while total > self.nTracks:
            if (total_12T-min_num_12T_tracks)*1.0/12 < (total_8T-min_num_8T_tracks)*1.0/8:
                flag = 0
            else:
                flag = 1
            candidate_value = []
            candidate_id = []
            for subid in range(len(row_label_order)):
                if row_label_order[subid] == flag:
                    candidate_value.append(row_label_num[subid])
                    candidate_id.append(subid)
            max_id = candidate_id[candidate_value.index(max(candidate_value))]
            #max_id = row_label_num.index(max(row_label_num))
            if row_label_order[max_id] == 0:
                row_label_num[max_id] = row_label_num[max_id] - 8
                total = total - 8
                total_8T = total_8T - 8
            else:
                row_label_num[max_id] = row_label_num[max_id] - 12
                total = total - 12
                total_12T = total_12T - 12
        
        print(total_12T, total_8T)
        #print(row_label_num)

        rowplacepermit = []
        rowLabelnew = []
        for id in range(len(row_label_num)):
            for subid in range(row_label_num[id]):
                if row_label_order[id] == 1:
                    rowLabelnew.append(12)
                else:
                    rowLabelnew.append(8)
                if row_label_order[id] == 1:
                    div = 12
                else:
                    div = 8
                if subid % div == 0:
                    rowplacepermit.append(1)
                else:
                    rowplacepermit.append(0)
        
        self.rowLabel = rowLabelnew
        self.rowplacepermit = rowplacepermit

        #for id in range(self.nRows):
        #    if first_order_id == 1:
        #        rowLabel[id] = 0
        #    else:
        #        rowLabel[id] = 1

        #if first_order_id == 1:
        #    for id in range(total_12T):
        #        rowLabel[distancemetric[id][0]] = 1
        #else:
        #    for id in range(total_8T):
        #        rowLabel[distancemetric[id][0]] = 0 

        #cnt_wrong = 0
        #cnt_total = 0
        #for id in range(len(self.rowplacepermit)):
        #    if self.rowplacepermit[id] == 1:
        #        cnt_total = cnt_total + 1
        #        if rowLabel[id] == 1:
        #            tmp = 12
        #        else:
        #            tmp = 8
        #        if self.rowLabel[id] != tmp:
        #            cnt_wrong = cnt_wrong + 1
        
        #print(cnt_total)
        #print(cnt_wrong)
        print(self.rowLabel)
        print(self.rowplacepermit)
    
    def PlaceCells(self):
        xcor_cells = []

        for id in range(len(self.vCells)):
            xcor_cells.append([id,self.vCells[id].locX])
        
        xcor_cells.sort(key=self.takeSecond)
        #print(xcor_cells)

        rowcell = {}
        rowcellwidth = {}
        rowrightmost = {}
        rowrightmostid = {}
        cellcor = []

        for id in range(len(self.vCells)):
            cellcor.append([0,0])

        for id in range(len(xcor_cells)):
            min_value = self.siteWidth*self.nSites+self.trackHeight*self.nTracks
            min_value_id = 0
            min_xcorr = 0
            mintotalcellwidth = 0
            Xrightmost = 0
            Xnewcorr = 0
            totalcellwidth = 0
            for row_id in range(len(self.rowplacepermit)):
                if self.rowplacepermit[row_id] == 0 or self.rowLabel[row_id] != self.vCells[xcor_cells[id][0]].height:
                    continue
                if rowrightmost.has_key(row_id):
                    Xrightmost = rowrightmost[row_id]
                else:
                    Xrightmost = 0
                
                if math.ceil(Xrightmost) < xcor_cells[id][1]:
                    Xnewcorr = math.ceil(xcor_cells[id][1])
                else:
                    Xnewcorr = math.ceil(Xrightmost)
                
                if rowcellwidth.has_key(row_id):
                    totalcellwidth = rowcellwidth[row_id]
                else:
                    totalcellwidth = 0

                if int(totalcellwidth + self.vCells[xcor_cells[id][0]].width) > self.nSites:
                    continue                                
                
                if self.siteWidth*abs(Xnewcorr-self.vCells[xcor_cells[id][0]].locX)+self.trackHeight*abs(row_id-self.vCells[xcor_cells[id][0]].locY)<min_value:
                    min_value = self.siteWidth*abs(Xnewcorr-self.vCells[xcor_cells[id][0]].locX)+self.trackHeight*abs(row_id-self.vCells[xcor_cells[id][0]].locY)
                    min_value_id = row_id
                    min_xcorr = Xnewcorr
                    mintotalcellwidth = totalcellwidth
            
            if rowrightmost.has_key(min_value_id):
                if min_xcorr + self.vCells[xcor_cells[id][0]].width > self.nSites:
                    for subid in range(len(rowcell[min_value_id])):
                        if subid == 0:
                            cellcor[rowcell[min_value_id][subid]][0] = 0
                        else:
                            cellcor[rowcell[min_value_id][subid]][0] = cellcor[rowcell[min_value_id][subid-1]][0] + self.vCells[rowcell[min_value_id][subid-1]].width
                    min_xcorr = cellcor[rowcell[min_value_id][-1]][0] + self.vCells[rowcell[min_value_id][-1]].width
                #print(min_xcorr)
                if len(rowcell[min_value_id]) != 0:
                    self.vCells[rowcell[min_value_id][-1]].laterCell = xcor_cells[id][0]
                    self.vCells[xcor_cells[id][0]].prevCell = rowcell[min_value_id][-1]
                rowcell[min_value_id].append(xcor_cells[id][0])
                rowrightmost[min_value_id] = min_xcorr + self.vCells[xcor_cells[id][0]].width            
                rowcellwidth[min_value_id] = mintotalcellwidth + self.vCells[xcor_cells[id][0]].width
            else:
                rowcelldict = {min_value_id:[xcor_cells[id][0]]}
                rowrightmostdict = {min_value_id:min_xcorr+self.vCells[xcor_cells[id][0]].width}
                rowcellwidthdict = {min_value_id:mintotalcellwidth+self.vCells[xcor_cells[id][0]].width}
                rowcell.update(rowcelldict)
                rowrightmost.update(rowrightmostdict)
                rowcellwidth.update(rowcellwidthdict)
            #print(min_xcorr, self.vCells[xcor_cells[id][0]].width)
            cellcor[xcor_cells[id][0]][0] = int(min_xcorr)
            cellcor[xcor_cells[id][0]][1] = min_value_id
        self.cellcor = cellcor
        #self.cellmatchingrefinement()
        return self.rowLabel, self.cellcor
    
    def cellmatchingrefinement(self):
        s = 30
        while(True):
            distancemetric = []
            mid_node = []
            sametypedis = []
            for id in range(self.nCells):
                distancemetric.append(self.siteWidth*abs(self.cellcor[id][0]-self.vCells[id].locX)+self.trackHeight*abs(self.cellcor[id][1]-self.vCells[id].locY))
            max_disp_old = max(distancemetric)
            min_id = distancemetric.index(max(distancemetric))
            if self.vCells[min_id].height == 12:
                min_type = 12
            else:
                min_type = 8
            mid_node.append((self.vCells[min_id].locX+self.cellcor[id][0])*1.0/2)
            mid_node.append((self.vCells[min_id].locY+self.cellcor[id][1])*1.0/2)
            for id in range(self.nCells):
                if self.vCells[id].height == min_type and id != min_id:
                    sametypedis.append([id, self.siteWidth*abs(self.vCells[id].locX - mid_node[0])+self.trackHeight*abs(self.vCells[id].locY - mid_node[1])])
            if len(sametypedis) > s:
                sametypedis.sort(key=takeSecond)
                sametypedis = sametypedis[0:s]
            sametypedis.append([min_id, 0])

            G = nx.Graph()

            edges = []
            for source_id in range(len(sametypedis)):
                for end_id in range(len(sametypedis)):
                    if self.vCells[sametypedis[end_id][0]].laterCell == -1:
                        right_bound = self.nSites
                    else:
                        right_bound = self.cellcor[self.vCells[sametypedis[end_id][0]].laterCell][0]
                    if self.vCells[sametypedis[source_id][0]].width <= right_bound - self.cellcor[sametypedis[end_id][0]][0] and self.vCells[sametypedis[end_id][0]].width <= right_bound - self.cellcor[sametypedis[source_id][0]][0]:
                        org_dis = self.siteWidth*abs(self.cellcor[sametypedis[source_id][0]][0]-self.vCells[sametypedis[source_id][0]].locX)+self.trackHeight*abs(self.cellcor[sametypedis[source_id][0]][1]-self.vCells[sametypedis[source_id][0]].locY)
                        new_dis = self.siteWidth*abs(self.cellcor[sametypedis[end_id][0]][0]-self.vCells[sametypedis[source_id][0]].locX)+self.trackHeight*abs(self.cellcor[sametypedis[end_id][0]][1]-self.vCells[sametypedis[source_id][0]].locY)
                        weight = org_dis - new_dis
                        print(sametypedis[source_id][0], sametypedis[end_id][0], weight)
                        edges.append((sametypedis[source_id][0],sametypedis[end_id][0],weight))
                        #G.add_edge(sametypedis[source_id][0],sametypedis[end_id][0],weight=weight)
            
            G.add_weighted_edges_from(edges)
            match = nx.max_weight_matching(G,maxcardinality=True)
            print(match)

            for subarr in match:
                tmp = self.cellcor[subarr[0]]
                self.cellcor[subarr[0]] = self.cellcor[subarr[1]]
                self.cellcor[subarr[1]] = tmp
            
            distancemetric = []
            for id in range(self.nCells):
                distancemetric.append(self.siteWidth*abs(self.cellcor[id][0]-self.vCells[id].locX)+self.trackHeight*abs(self.cellcor[id][1]-self.vCells[id].locY))
            
            max_disp_new = max(distancemetric)

            if max_disp_new >= max_disp_old:
                break                        

    def run(self):
        print("Row Assignment (8T or 12T)")
        self.KMeansRowHeightAssignment()
        print("Place Cells")
        return self.PlaceCells()


class SolutionGenerator:
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.nSites = None
        self.nRows = None
        self.nTracks = None

        self.siteWidth = 0.216
        self.trackHeight = 0.27
        
        self.nCells = 0
        self.vCells = []
        self.name2CellId = {}
        
        self.vLegCells = []
        self.name2LegCellId = {}

        self.displBound = None
    
    def readInput(self, path):
        with open(path, "r") as f:
            data = f.read().splitlines()

        for i, line in enumerate(data):
            tokens = line.strip().split(" ")
            tokens = [ elem.strip() for elem in tokens ]

            if i == 0:
                self.nRows, self.nSites = int(tokens[0]), int(tokens[1])
            elif i == 1:
                self.trackHeight, self.siteWidth = float(tokens[0]), float(tokens[1])
            elif i == 2:
                self.displBound = float(tokens[0])
            elif i == 3:
                self.nCells = int(tokens[0])
            else:
                name = tokens[0]
                locY, locX = float(tokens[1]), float(tokens[2])
                h, w = int(tokens[3]), int(tokens[4])
                cell = Cell(locX, locY, w, h)
                self.name2CellId[len(self.vCells)] = name
                self.vCells.append(cell)
        assert len(self.vCells) == self.nCells
        assert len(self.vCells) == len(self.name2CellId)
        self.nTracks = 4 * self.nRows
    
    def detailplaceCells(self):
        self.detailplacer = DetailPlacer(self.nRows, self.nTracks, self.nSites, self.nCells, self.vCells, self.name2CellId, self.trackHeight, self.siteWidth)
        self.rowlabel, self.Placecellcor = self.detailplacer.run()
    
    def ResultOutput(self, path):
        row_start = []
        row_label_name = []

        id = 0

        while id < len(self.rowlabel):
            row_start.append(id)
            row_label_name.append(self.rowlabel[id])
            for iter in range(self.rowlabel[id]):
                id = id + 1
        
        outstr = ''
        outstr += str(len(row_start))
        outstr += '\n'

        for id in range(len(row_start)):
            outstr += str(row_start[id])
            outstr += ' '
            outstr += str(row_label_name[id])
            outstr += '\n'
        
        for id in range(len(self.Placecellcor)):
            outstr += self.name2CellId[id]
            outstr += ' '
            outstr += str(int(self.Placecellcor[id][1]))
            outstr += ' '
            outstr += str(int(self.Placecellcor[id][0]))
            outstr += '\n'
        
        fo = open(path, "w")
        fo.write(outstr)
        fo.close()
    
    def run(self, in_file, out_file):
        self.readInput(in_file)
        self.detailplaceCells()
        self.ResultOutput(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    evaluator = SolutionGenerator()
    evaluator.run(args.input, args.output)
                




    
    
    
    
