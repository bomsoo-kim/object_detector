# %%
import numpy as np
import heapq
import matplotlib.pyplot as plt

def group_text_into_line(list_text_xyxy, alpha=1.0, debug=False):
    """
    list_text_xyxy = [ (text, (x0, y0, x1, y1)), ... ]
    """
    uid_to_word_map = {}
    for uid, (text, (x0,y0,x1,y1)) in enumerate(list_text_xyxy):
        uid_to_word_map[uid] = (text, (x0,y0,x1,y1))

    #================================================================
    seqs = {} # text sequences
    for uid in uid_to_word_map.keys():
        text, (x0, y0, x1, y1) = uid_to_word_map[uid]
        cx = 0.5*(x0 + x1)
        dw = (x1 - x0) / len(text) * alpha
        x0, y0, x1, y1 = x0-dw, y0, x1+dw, y1
        if debug:
            print(f'uid:text = {uid},:{text}')
        
        attach_to_head, attach_to_tail = None, None
        #--- start to check for each text group ------------------------------
        for uid0, node0 in seqs.items(): # O(N^2) in the worse case
            first_uid = node0['uid']
            last_uid = node0['last']['uid']
    
            #-------------------------------------------------
            text_f, (x0_f, y0_f, x1_f, y1_f) = uid_to_word_map[first_uid]
            cx_f = 0.5*(x0_f + x1_f)
            dw_f = (x1_f - x0_f) / len(text_f) * alpha
            x0_f, y0_f, x1_f, y1_f = x0_f-dw_f, y0_f, x1_f+dw_f, y1_f
            
            intersection_area_f = max(0, min(x1, x1_f) - max(x0, x0_f)) * max(0, min(y1, y1_f) - max(y0, y0_f))
            
            if intersection_area_f > 0:
                if (first_uid != last_uid):
                    attach_to_head = uid0
                else:
                    if cx <= cx_f:
                        attach_to_head = uid0
                    else:
                        attach_to_tail = uid0

            #-------------------------------------------------
            text_l, (x0_l, y0_l, x1_l, y1_l) = uid_to_word_map[last_uid]
            dw_l = (x1_l - x0_l) / len(text_l) * alpha
            x0_l, y0_l, x1_l, y1_l = x0_l-dw_l, y0_l, x1_l+dw_l, y1_l

            intersection_area_l = max(0, min(x1, x1_l) - max(x0, x0_l)) * max(0, min(y1, y1_l) - max(y0, y0_l))

            if (first_uid != last_uid) and (intersection_area_l > 0):
                attach_to_tail = uid0

            if debug:
                print(f'> uid0:text = {first_uid}:{text_f}, uid1:text = {last_uid}:{text_l}')
                print(f'> attach_to_head = {attach_to_head}, attach_to_tail = {attach_to_tail}')

            
            if (attach_to_head is not None) and (attach_to_tail is not None):
                assert attach_to_head != attach_to_tail
                break

        #-------------------------------------------------
        if attach_to_head is not None:
            node0 = seqs[attach_to_head]

            new = {'uid':uid} # create a new node
            new['next'] = node0 # now, the new node is the first node
            new['last'] = node0['last'] # the first node is responsible for keeping the information about 'last'
            del node0['last'] # this is no longer the first node and so don't need to keep the information about 'last'
            
            seqs[uid] = new
            del seqs[attach_to_head]
            
        #-------------------------------------------------
        if attach_to_tail is not None:
            node0 = seqs[attach_to_tail]

            if uid not in seqs:
                new = {'uid':uid, 'next':None} # create a new node
                node0['last']['next'] = new
                node0['last'] = new
                if debug:
                    print(f'> attach_to_tail --> scenario 1')
            else:
                old = seqs[uid]
                node0['last']['next'] = old
                node0['last'] = old['last']
                del seqs[uid]
                if debug:
                    print(f'> attach_to_tail --> scenario 2')

        #--- if there is no overlap, then create a new seqs item -----------------------
        if (attach_to_head is None) and (attach_to_tail is None):
            new = {'uid':uid, 'next':None}
            new['last'] = new # the last node is itself, because there is only one node
            seqs[uid] = new

    #=======================================================================
    i_to_line_map = {}
    for i, (uid, node) in enumerate(seqs.items()):
        assert uid == node['uid']
        arr = []
        x0_min = float('inf')
        y0_min = float('inf')
        x1_max = float('-inf')
        y1_max = float('-inf')
        
        while node is not None:
            # print(t, end=' ::: ')
            t, (x0,y0,x1,y1) = uid_to_word_map[node['uid']]
            arr.append(t)
            x0_min = min(x0_min, x0)
            y0_min = min(y0_min, y0)
            x1_max = max(x1_max, x1)
            y1_max = max(y1_max, y1)
            
            node = node['next']
        # print(f' <<{i+1}>> ')
        
        line_data = (' '.join(arr), (x0_min, y0_min, x1_max, y1_max))
        i_to_line_map[i] = line_data

    return i_to_line_map, seqs, uid_to_word_map

def find_single_passage(seqs, uid_to_word_map, delimiter='\t'):
    def find_x_center(node):
        _, (x0, _, _, _) = uid_to_word_map[node['uid']] # first node of a sequence
        _, (_, _, x1, _) = uid_to_word_map[node['last']['uid']] # last node of a sequence
        return 0.5*(x0 + x1)
        
    #-------------------------------------------------------------------
    longlines = {}
    for i, node in enumerate(seqs.values()):
        _, (_, y0_f, _, y1_f) = uid_to_word_map[node['uid']] # first node of a sequence
        _, (_, y0_l, _, y1_l) = uid_to_word_map[node['last']['uid']] # last node of a sequence
        y0 = min(y0_f, y0_l)
        y1 = max(y1_f, y1_l)
        
        found = False
        for lline in longlines.values(): # O(n^2) time complexity in the worst case
            yy0 = lline['y0']
            yy1 = lline['y1']
            
            if max(y0, yy0) < min(y1, yy1): # if the lines are overlaping 
                lline['stack'].append(node) # this que will be sorted later
                lline['y0'] = min(y0, yy0) # update x0
                lline['y1'] = max(y1, yy1) # update x1
                found = True
                break # if found, then exit from the for loop, because there is no need to find any more
            
        if not found:
            longlines[i] = {'stack':[node], 'y0':y0, 'y1':y1} # initialization

    #-------------------------------------------------------------------
    passage = []
    for lline in longlines.values(): # sort in x direction
        stack_sorted = sorted(lline['stack'], key=lambda node: find_x_center(node))
        
        sentences, x0, x1 = [], float('inf'), float('-inf')
        for node in stack_sorted:
            sentence = []
            while node:
                text, (xx0, yy0, xx1, yy1) = uid_to_word_map[node['uid']]
                sentence.append(text)
                x0 = min(x0, xx0)
                x1 = max(x1, xx1)
                node = node['next']
            sentences.append(' '.join(sentence))
        passage.append([delimiter.join(sentences), (x0, lline['y0'], x1, lline['y1'])])
    
    passage = sorted(passage, key=lambda s: 0.5*(s[-1][1] + s[-1][3])) # 0.5*(y0 + y1) sort in x direction

    return passage, longlines


def find_index_left_right_most(idx_source, i_to_line_map):
    """
    To find the left or right most text, heap queue is used.
    """
    _, (x0,y0,x1,y1) = i_to_line_map[idx_source]
    cx = 0.5*(x0 + x1)
    cy = 0.5*(y0 + y1)
    
    left, right = [], [] # don't neet to do 'heapq.heapify()', because it's empty
    for j, (_, (xx0,yy0,xx1,yy1)) in i_to_line_map.items():
        if (j != idx_source) and (yy0 <= cy <= yy1):
            cxx = 0.5*(xx0 + xx1)
            if cx < cxx:
                heapq.heappush(right, (cxx, j))
            elif cxx < cx:
                heapq.heappush(left, (-cxx, j))
            else:
                raise(Exception('Brad error: unknown situation encountered...'))

    idx_left = left[0][1] if left else None
    idx_right = right[0][1] if right else None
    
    return {'idx_left':idx_left, 'idx_right':idx_right}

### TEST ################################################################################
if __name__=='__main__':
    from PIL import Image
    from faster_rcnn import plot_text_xyxy

    img = np.array(Image.open(r'D:\codes\OCR\metrics_team2.JPG'))

    # out = {'pred_texts': [('PERMANENT', (95, 610, 187, 623)),.......


    list_text_xyxy = out['pred_texts']

    i_to_line_map, seqs, uid_to_word_map = group_text_into_line(list_text_xyxy, alpha=1.0, debug=False)
    passage, longlines = find_single_passage(seqs, uid_to_word_map, delimiter='\t')
    # passage, longlines = find_single_passage(seqs, uid_to_word_map, delimiter=';')

    print('\n'.join(p[0] for p in passage)) # text extracted

    plot_text_xyxy(img, i_to_line_map.values())

    plot_text_xyxy(img, passage)


# if __name__ == "__main__":
#     from brad_text import Word_Source_Dictionary

#     wsd = Word_Source_Dictionary()

#     for source, (text, _)  in i_to_line_map.items():
#         wsd.add(text, source) # create word-source dictionary

#     # print(wsd.find_sources(['venezuela'])) # 'apple' AND 'banana'
#     # print(wsd.find_sources(['apple', 'door'])) # 'apple' OR 'door'
#     # print(wsd.find_sources(['apple door'])) # 'apple' AND 'door'

#     # print(wsd.find_sources(['pp']))
#     # print(wsd.find_sources(['app'], allow_prefix=False, allow_suffix=False)) # exact matching
#     # print(wsd.find_sources(['apple']))
#     # print(wsd.find_sources(['apple'], allow_prefix=False, allow_suffix=False, allow_inner_suffix=True, allow_inner_prefix=True))
#     # print(wsd.find_sources(['apple'], allow_prefix=True, allow_suffix=True, allow_inner_suffix=True, allow_inner_prefix=True))

#     ii = wsd.find_sources(['china'])
#     # ii = wsd.find_sources(['US'], allow_prefix=False, allow_suffix=False) # exact matching
#     for i in ii:
#         print(i, i_to_line_map[i])


