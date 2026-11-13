"""Predeclared continuous start/stop and signed-turn reversal protocol."""
SCHEDULE=[(0,3,[0,0,0],'stand'),(3,8,[.2,0,0],'start 0.2'),
          (8,13,[.4,0,0],'accelerate 0.4'),(13,17,[0,0,0],'brake stop'),
          (17,23,[.2,0,.5],'left turn'),(23,29,[.2,0,-.5],'reverse right'),
          (29,35,[.2,0,.5],'reverse left'),(35,40,[.2,0,0],'straighten'),
          (40,46,[0,0,0],'final stop')]


def command_at(t):
    import numpy as np
    for start,end,cmd,label in SCHEDULE:
        if start<=t<end:return np.array(cmd,dtype=float),label
    return np.zeros(3),'final stop'
