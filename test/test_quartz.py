import Quartz

def MyFunction(p, t, e, c):
    print e

tap = Quartz.CGEventTapCreate(Quartz.kCGHIDEventTap, Quartz.kCGHeadInsertEventTap, Quartz.kCGEventTapOptionListenOnly, Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown), MyFunction, None)

runLoopSource = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0);
Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), runLoopSource, Quartz.kCFRunLoopDefaultMode);
Quartz.CGEventTapEnable(tap, True);

Quartz.CFRunLoopRun();
