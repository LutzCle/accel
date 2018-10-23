use error::*;
use ffi::cudart::*;
use std::mem;
use std::ops;
use std::os::raw::c_uint;
use std::ptr::null_mut;

pub struct Event(cudaEvent_t);

impl Event {
    pub fn new() -> Result<Self> {
        let mut event: cudaEvent_t = unsafe { mem::zeroed() };
        unsafe { cudaEventCreate(&mut event) }.check()?;
        Ok(Event(event))
    }

    pub fn record(&self) -> Result<&Self> {
        // FIXME: using default stream
        unsafe { cudaEventRecord(self.0, null_mut()) }.check()?;
        Ok(self)
    }

    pub fn synchronize(&self) -> Result<&Self> {
        unsafe { cudaEventSynchronize(self.0) }.check()?;
        Ok(self)
    }

    pub fn elapsed_time(&self, start: &Self) -> Result<f32> {
        let mut millis: f32 = 0.0;
        unsafe { cudaEventElapsedTime(&mut millis, start.0, self.0) }.check()?;
        Ok(millis)
    }
}

impl ops::Drop for Event {
    fn drop(&mut self) {
        unsafe { cudaEventDestroy(self.0) }
            .check()
            .expect("Couldn't destroy CUDA event");
    }
}

pub struct EventWithOptions {
    blocking_sync: bool,
    disable_timing: bool,
    interprocess: bool,
}

impl EventWithOptions {
    pub fn default() -> Self {
        Self {
            blocking_sync: false,
            disable_timing: false,
            interprocess: false,
        }
    }

    pub fn blocking_sync(&mut self, option: bool) -> &mut Self {
        self.blocking_sync = option;
        self
    }

    pub fn disable_timing(&mut self, option: bool) -> &mut Self {
        self.disable_timing = option;
        self
    }

    pub fn interprocess(&mut self, option: bool) -> &mut Self {
        self.interprocess = option;
        self
    }

    pub fn create(&mut self) -> Result<Event> {
        let mut event: cudaEvent_t = unsafe { mem::zeroed() };

        let flags: c_uint = if self.blocking_sync {
            cudaEventBlockingSync
        } else {
            cudaEventDefault
        } | if self.disable_timing {
            cudaEventDisableTiming
        } else {
            cudaEventDefault
        } | if self.interprocess {
            cudaEventInterprocess
        } else {
            cudaEventDefault
        };

        unsafe { cudaEventCreateWithFlags(&mut event, flags) }.check()?;
        Ok(Event(event))
    }
}
