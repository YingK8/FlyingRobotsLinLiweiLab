import rp2
from machine import Pin, mem32
import uctypes
import array

# ==========================================
# 1. Constants & Configuration
# ==========================================

# Hardware Base Addresses (RP2040)
DMA_BASE      = 0x50000000
PIO0_BASE     = 0x50200000

# DMA Register Offsets
DMA_READ_ADDR = 0x00
DMA_WRITE_ADDR= 0x04
DMA_TRANS_CNT = 0x08
DMA_TRIG_CTRL = 0x0c
DMA_CTRL      = 0x10

# DMA Configuration Flags
DMA_ENABLE    = 1 << 0
DMA_HIGH_PRIO = 1 << 2
DMA_SIZE_32   = 2 << 2
DMA_INC_READ  = 1 << 4
DMA_INC_WRITE = 1 << 5
DMA_CHAIN_TO  = 21  # Bit offset for chaining
DMA_TREQ_SEL  = 15  # Bit offset for transfer request
TREQ_PERMANENT= 0x3F # Always ready

# PWM Configuration
PWM_CHANNELS  = 10
CLOCK_FREQ    = 125_000_000 # 125 MHz

# ==========================================
# 2. PIO Assembly Program
# ==========================================
# This program handles the timing. It expects 32-bit commands:
# - Lower 10 bits: The state of the 10 pins (1=High, 0=Low)
# - Upper 22 bits: How many cycles to wait before the next command
@rp2.asm_pio(out_init=(rp2.PIO.OUT_LOW,) * 10, out_shiftdir=rp2.PIO.SHIFT_RIGHT, autopull=True, pull_thresh=32)
def pwm_prog():
    out(pins, 10)       # 1. Set the 10 pins to the values in the lower 10 bits
    out(x, 22)          # 2. Read the delay duration (upper 22 bits) into register X
    label("delay_loop")
    jmp(x_dec, "delay_loop") # 3. Count down X until it reaches 0 (1 cycle per loop)

# ==========================================
# 3. PWM Controller Class
# ==========================================
class PhasePWM:
    def __init__(self, base_pin, freq=1000, sm_id=0, dma_channel=0):
        self.freq = freq
        self.sm_id = sm_id
        
        # We need two DMA channels for an infinite loop (Ping-Pong)
        # Channel A: Sends data to PIO
        # Channel B: Reconfigures Channel A to start again
        self.dma_data = dma_channel
        self.dma_ctrl = dma_channel + 1
        
        # Storage for Phase (0-360) and Duty (0-100)
        self.phases = [0.0] * PWM_CHANNELS
        self.duties = [0.0] * PWM_CHANNELS
        
        # Initialize GPIO pins
        self.pins = [Pin(base_pin + i, Pin.OUT) for i in range(PWM_CHANNELS)]

        # Initialize PIO State Machine
        self.sm = rp2.StateMachine(self.sm_id, pwm_prog, freq=CLOCK_FREQ, 
                                   out_base=self.pins[0], set_base=self.pins[0])
        self.sm.active(1)

        # Pre-allocate memory buffers (Double buffering strategy)
        # 3 events per channel is plenty (Start, Stop, Wrap)
        buffer_size = PWM_CHANNELS * 4 
        self.buffer_A = array.array('I', [0] * buffer_size)
        self.buffer_B = array.array('I', [0] * buffer_size)
        self.active_buffer = self.buffer_A

        # Calculate PIO TX FIFO Address
        self.pio_tx_addr = PIO0_BASE + 0x10 + (self.sm_id * 4)

        # Start the system
        self.update()
        self._start_dma_loop()

    def set_config(self, channel, phase, duty):
        """ Update a single channel's settings """
        self.phases[channel] = phase % 360.0
        self.duties[channel] = max(0.0, min(100.0, duty))
        self.update()

    def update(self):
        """ 
        The Core Logic: Converts Phase/Duty % into a timeline of 
        ON/OFF commands for the PIO.
        """
        # 1. Calculate the total clock ticks in one PWM period
        period_ticks = int(CLOCK_FREQ / self.freq)
        
        # 2. Create a list of "events" where pins need to change
        # Format: (Tick Time, Channel Index, New State)
        events = []
        
        # Add events for each channel
        for ch in range(PWM_CHANNELS):
            duty = self.duties[ch]
            phase = self.phases[ch]
            
            # Skip always-off or always-on cases (handled later by logic)
            if duty <= 0 or duty >= 100: continue
            
            # Convert Phase & Duty to Clock Ticks
            start_tick = int((phase / 360.0) * period_ticks)
            on_duration = int((duty / 100.0) * period_ticks)
            end_tick = (start_tick + on_duration) % period_ticks
            
            events.append((start_tick, ch, 1)) # Turn ON
            events.append((end_tick, ch, 0))   # Turn OFF
            
        # 3. Sort events by time (so we process them in order)
        events.sort(key=lambda x: x[0])
        
        # 4. Generate the machine code for PIO
        # We write into the "inactive" buffer to avoid glitches
        target_buffer = self.buffer_B if self.active_buffer == self.buffer_A else self.buffer_A
        
        current_time = 0
        # Determine initial state of all pins at Time=0
        current_mask = 0
        for ch in range(PWM_CHANNELS):
            p = self.phases[ch]
            d = self.duties[ch]
            # Check if this channel is active at time 0 (wraps around)
            if d >= 100: current_mask |= (1 << ch)
            elif d > 0:
                p_end = p + (d * 3.6) # Convert duty % to degrees
                if p_end > 360: current_mask |= (1 << ch)

        # Iterate through sorted events
        buf_idx = 0
        i = 0
        while i < len(events):
            tick = events[i][0]
            
            # Combine all pin changes happening at this exact same tick
            while i < len(events) and events[i][0] == tick:
                ch = events[i][1]
                state = events[i][2]
                if state == 1:
                    current_mask |= (1 << ch) # Set bit
                else:
                    current_mask &= ~(1 << ch) # Clear bit
                i += 1
            
            # Calculate delay since last event
            delta = tick - current_time
            
            # Subtract overhead (approx 3 cycles for PIO instructions)
            # If delta is tiny, we just send immediately (0 delay)
            delay_cycles = max(0, delta - 3)
            
            if delta > 0:
                # Pack Data: [22 bits Delay | 10 bits Pin Mask]
                command = (delay_cycles << 10) | current_mask
                target_buffer[buf_idx] = command
                buf_idx += 1
                current_time = tick

        # 5. Fill the time until end of period
        remaining = period_ticks - current_time
        if remaining > 3:
            delay = remaining - 3
            command = (delay << 10) | current_mask
            target_buffer[buf_idx] = command
            buf_idx += 1
            
        # 6. Mark this buffer as the active one for the next DMA loop
        self.active_buffer = target_buffer
        self.active_len = buf_idx

        # If DMA is already running, we just need to update the source address
        # for the "Control" channel.
        # This is safe because we are just updating a pointer in RAM.
        if hasattr(self, 'addr_pointer'):
            self.addr_pointer[0] = uctypes.addressof(target_buffer)

    def _start_dma_loop(self):
        """ 
        Sets up the DMA 'Infinite Loop'. 
        Uses raw register access for speed/control. 
        """
        # Pointer to the address of the buffer (Double pointer)
        # The Control DMA reads this to find out where the Data buffer is
        self.addr_pointer = array.array('I', [uctypes.addressof(self.active_buffer)])

        # --- Configure Control Channel (DMA B) ---
        # Job: Copy 'addr_pointer' -> 'DMA A Read Address'
        ctrl_config = (
            DMA_ENABLE | 
            DMA_HIGH_PRIO | 
            DMA_SIZE_32 |
            TREQ_PERMANENT << DMA_TREQ_SEL |
            (self.dma_data << DMA_CHAIN_TO) # Chain back to Data Channel
        )
        
        # Write Registers for Control Channel
        offset_ctrl = self.dma_ctrl * 0x40
        mem32[DMA_BASE + offset_ctrl + DMA_READ_ADDR] = uctypes.addressof(self.addr_pointer)
        mem32[DMA_BASE + offset_ctrl + DMA_WRITE_ADDR]= DMA_BASE + (self.dma_data * 0x40) + DMA_READ_ADDR
        mem32[DMA_BASE + offset_ctrl + DMA_TRANS_CNT] = 1
        mem32[DMA_BASE + offset_ctrl + DMA_CTRL]      = ctrl_config

        # --- Configure Data Channel (DMA A) ---
        # Job: Copy 'active_buffer' -> 'PIO TX FIFO'
        # DREQ (Data Request) depends on which State Machine (SM) we use.
        # For SM0, DREQ=0. For SM1, DREQ=1.
        dreq = self.sm_id 
        
        data_config = (
            DMA_ENABLE |
            DMA_HIGH_PRIO |
            DMA_SIZE_32 |
            DMA_INC_READ | # Increment read address (walk through buffer)
            (dreq << DMA_TREQ_SEL) | 
            (self.dma_ctrl << DMA_CHAIN_TO) # Chain to Control Channel when done
        )

        # Write Registers for Data Channel
        offset_data = self.dma_data * 0x40
        mem32[DMA_BASE + offset_data + DMA_READ_ADDR] = uctypes.addressof(self.active_buffer)
        mem32[DMA_BASE + offset_data + DMA_WRITE_ADDR]= self.pio_tx_addr
        mem32[DMA_BASE + offset_data + DMA_TRANS_CNT] = self.active_len
        mem32[DMA_BASE + offset_data + DMA_TRIG_CTRL] = data_config # Writing TRIG starts it!