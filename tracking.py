import localization

def localize_frame(data, frame, params):
    print('localizing frame {0}'.format(frame), end = "\r", flush = True)
    bubbles = localization.Localize(frame = frame)
    bubbles.localize_bubbles(data[:,:,frame],params)
    bubbles.merge_coincident_bubbles(params.N_merge)
    bubbles.refine_centers(data[:,:,frame],params)
    bubbles.filter_candidates(data[:,:,frame],params)
    #bubbles.get_bubble_widths(img[:,:,frame],P)
    return bubbles
