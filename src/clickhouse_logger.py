"""
ClickHouse Logger for RL Training
Provides simple functions to log step-level training data to ClickHouse.

Usage:
        logger = ClickHouseLogger(
            host='localhost',
            port=9000,
            database='rl_training',
            table_name='training_logs'
        )
        logger.initialize()
        
        # Log step-level data
        logger.log_step(
            episode=1,
            timestep=100,
            reward=0.5,
            avg_reward=0.45
        )
        
        # Log custom metrics
        logger.log_step(
            episode=1,
            timestep=101,
            reward=0.6,
            avg_reward=0.46,
            action_std=0.5,
            loss=0.01
        )
        
        logger.close()
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
import os
from clickhouse_connect import get_client

class ClickHouseLogger:
    """
    A simple logger class for storing step-level training data in ClickHouse.
    """
    
    def __init__(self):
        self.table_name = 'training_logs'

        self.batch_size = 100
        self.batch_buffer = []

        self.client = get_client(
            host=os.getenv('CLICKHOUSE_HOST'),
            port=int(os.getenv('CLICKHOUSE_PORT')),
            user=os.getenv('CLICKHOUSE_USER'),
            password=os.getenv('CLICKHOUSE_PASSWORD'),
            secure=True
        )
        
        # Test connection
        try:
            self.client.command('SELECT 1')
        except Exception as e:
            logging.error(f"Failed to connect to ClickHouse: {e}")
            raise
        self.create_table()

        logging.info(f"ClickHouse logger initialized.")
    
    def create_table(self) -> None:
        """Create the training logs table if it doesn't exist."""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name}
        (
            timestamp DateTime DEFAULT now(),
            episode UInt32,
            timestep UInt64,
            reward Float32,
            avg_reward Nullable(Float32),
            action_std Nullable(Float32),
            loss Nullable(Float32),
            value_estimate Nullable(Float32),
            entropy Nullable(Float32),
            policy_loss Nullable(Float32),
            value_loss Nullable(Float32),
            env_name String DEFAULT '',
            run_id String DEFAULT '',
            metadata String DEFAULT ''
        )
        ENGINE = MergeTree()
        ORDER BY (episode, timestep)
        """
        
        try:
            self.client.command(create_table_query)
            logging.info(f"Table {self.table_name} created or already exists")
        except Exception as e:
            logging.error(f"Failed to create table: {e}")
            raise
    
    def log_step(
        self,
        episode: int,
        timestep: int,
        reward: float,
        avg_reward: Optional[float] = None,
        action_std: Optional[float] = None,
        loss: Optional[float] = None,
        value_estimate: Optional[float] = None,
        entropy: Optional[float] = None,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        env_name: str = '',
        run_id: str = '',
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log step-level training data.
        
        Args:
            episode: Episode number
            timestep: Current timestep
            reward: Reward for this step
            avg_reward: Average reward (optional)
            action_std: Action standard deviation (optional)
            loss: Total loss (optional)
            value_estimate: Value function estimate (optional)
            entropy: Policy entropy (optional)
            policy_loss: Policy loss (optional)
            value_loss: Value loss (optional)
            env_name: Environment name (optional)
            run_id: Run identifier (optional)
            metadata: Additional metadata as dict (will be JSON stringified)
        """

        # Convert metadata dict to JSON string if provided
        metadata_str = ''
        if metadata:
            import json
            metadata_str = json.dumps(metadata)
        
        data = {
            'episode': episode,
            'timestep': timestep,
            'reward': float(reward),
            'avg_reward': float(avg_reward) if avg_reward is not None else None,
            'action_std': float(action_std) if action_std is not None else None,
            'loss': float(loss) if loss is not None else None,
            'value_estimate': float(value_estimate) if value_estimate is not None else None,
            'entropy': float(entropy) if entropy is not None else None,
            'policy_loss': float(policy_loss) if policy_loss is not None else None,
            'value_loss': float(value_loss) if value_loss is not None else None,
            'env_name': str(env_name),
            'run_id': str(run_id),
            'metadata': metadata_str,
        }
    
        self.batch_buffer.append(data)
        if len(self.batch_buffer) >= self.batch_size:
            self._flush_batch()
    
    def _flush_batch(self) -> None:
        """Flush batched records to ClickHouse."""
        if not self.batch_buffer:
            return
        
        try:
            column_names = [
                'episode', 'timestep', 'reward', 'avg_reward', 'action_std', 'loss',
                'value_estimate', 'entropy', 'policy_loss', 'value_loss', 'env_name', 'run_id', 'metadata'
            ]
            
            values_list = [
                [
                    d['episode'],
                    d['timestep'],
                    d['reward'],
                    d['avg_reward'],
                    d['action_std'],
                    d['loss'],
                    d['value_estimate'],
                    d['entropy'],
                    d['policy_loss'],
                    d['value_loss'],
                    d['env_name'],
                    d['run_id'],
                    d['metadata'],
                ]
                for d in self.batch_buffer
            ]
            
            self.client.insert(
                table=self.table_name,
                data=values_list,
                column_names=column_names
            )
            self.batch_buffer.clear()
            
        except Exception as e:
            logging.error(f"Failed to flush batch: {e}")
            # Don't raise - allow training to continue even if logging fails
    
    def flush(self) -> None:
        """Manually flush any buffered records."""
        if self.batch_buffer:
            self._flush_batch()
    
    def close(self) -> None:
        """Close the connection and flush any remaining data."""
        if self.batch_buffer:
            self._flush_batch()

        if self.client:
            self.client.disconnect()
            logging.info("ClickHouse logger closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass

