import { describe, it, expect, jest } from '@jest/globals';
import { progressTracker, ProgressUpdate } from '../utils/progress.js';

describe('ProgressTracker', () => {
  describe('subscribe and report', () => {
    it('should allow subscribing to progress updates', () => {
      const callback = jest.fn();
      progressTracker.subscribe('test-op-1', callback);

      const update: ProgressUpdate = {
        stage: 'test',
        progress: 0.5,
        message: 'Test progress'
      };

      progressTracker.report('test-op-1', update);
      expect(callback).toHaveBeenCalledWith(update);

      progressTracker.unsubscribe('test-op-1', callback);
    });

    it('should support multiple callbacks', () => {
      const callback1 = jest.fn();
      const callback2 = jest.fn();

      progressTracker.subscribe('test-op-2', callback1);
      progressTracker.subscribe('test-op-2', callback2);

      const update: ProgressUpdate = {
        stage: 'test',
        progress: 0.5,
        message: 'Test'
      };

      progressTracker.report('test-op-2', update);

      expect(callback1).toHaveBeenCalledWith(update);
      expect(callback2).toHaveBeenCalledWith(update);

      progressTracker.unsubscribe('test-op-2', callback1);
      progressTracker.unsubscribe('test-op-2', callback2);
    });

    it('should include optional fields', () => {
      const callback = jest.fn();
      progressTracker.subscribe('test-op-3', callback);

      progressTracker.report('test-op-3', {
        stage: 'processing',
        progress: 0.5,
        message: 'Halfway',
        eta: 120,
        details: { itemsProcessed: 50 }
      });

      const call = callback.mock.calls[0][0] as ProgressUpdate;
      expect(call.eta).toBe(120);
      expect(call.details).toEqual({ itemsProcessed: 50 });

      progressTracker.unsubscribe('test-op-3', callback);
    });
  });
});
