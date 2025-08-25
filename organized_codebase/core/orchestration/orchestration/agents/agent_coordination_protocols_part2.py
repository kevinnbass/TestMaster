"""
                
            except Exception as e:
                self.logger.error(f"Error in consensus protocol: {e}")
                time.sleep(1)
    
    def _heartbeat_loop(self):
        """Heartbeat management loop"""
        while self.coordination_active:
            try:
                current_time = datetime.now()
                
                # Send heartbeats to all cluster members
                for member_id in self.cluster_members:
                    if member_id != self.agent_id:
                        self.send_coordination_message(
                            target_agent_id=member_id,
                            protocol_type=ProtocolType.HEARTBEAT,
                            message_type="heartbeat",
                            payload={
                                "sender_id": self.agent_id,
                                "term": self.current_term,
                                "role": self.current_role.value,
                                "timestamp": current_time.isoformat()
                            },
                            priority=1,
                            ttl_seconds=30
                        )
                
                # Check for failed agents
                timeout_threshold = timedelta(seconds=60)
                for agent_id, state in self.cluster_members.items():
                    if (current_time - state.last_heartbeat > timeout_threshold and
                        state.status == "active"):
                        state.status = "failed"
                        self.logger.warning(f"Agent failed (no heartbeat): {agent_id}")
                        self._handle_agent_failure(agent_id)
                
                time.sleep(self.protocol_config["consensus"]["heartbeat_interval_ms"] / 1000)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(5)
    
    def _load_monitoring_loop(self):
        """Load monitoring and balancing loop"""
        while self.coordination_active:
            try:
                # Collect current load metrics
                self._collect_load_metrics()
                
                # Update routing tables
                self._update_all_routing_tables()
                
                # Trigger failover if needed
                self._check_failover_conditions()
                
                time.sleep(self.protocol_config["load_balancing"]["load_update_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in load monitoring: {e}")
                time.sleep(30)
    
    def _security_monitoring_loop(self):
        """Security monitoring loop"""
        while self.coordination_active:
            try:
                # Refresh authentication tokens
                self._refresh_auth_tokens()
                
                # Monitor for security threats
                self._detect_security_anomalies()
                
                # Update trusted agent list
                self._update_trusted_agents()
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in security monitoring: {e}")
                time.sleep(600)
    
    def _protocol_optimization_loop(self):
        """Protocol optimization loop"""
        while self.coordination_active:
            try:
                if self.protocol_config["monitoring"]["protocol_optimization"]:
                    self._optimize_protocol_parameters()
                    self._analyze_protocol_performance()
                
                time.sleep(600)  # Optimize every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in protocol optimization: {e}")
                time.sleep(1200)
    
    # Message handlers
    def _handle_vote_request(self, message: CoordinationMessage):
        """Handle vote request in leader election"""
        payload = message.payload
        
        # Check if we should grant vote
        grant_vote = (
            payload["term"] > self.current_term or
            (payload["term"] == self.current_term and 
             (self.voted_for is None or self.voted_for == message.source_agent_id))
        )
        
        if grant_vote:
            self.current_term = payload["term"]
            self.voted_for = message.source_agent_id
            self.current_role = AgentRole.FOLLOWER
        
        # Send vote response
        self.send_coordination_message(
            target_agent_id=message.source_agent_id,
            protocol_type=ProtocolType.CONSENSUS,
            message_type="vote_response",
            payload={
                "term": self.current_term,
                "vote_granted": grant_vote,
                "voter_id": self.agent_id
            },
            priority=9
        )
    
    def _handle_vote_response(self, message: CoordinationMessage):
        """Handle vote response in leader election"""
        if self.current_role != AgentRole.CANDIDATE:
            return
        
        payload = message.payload
        
        if payload["vote_granted"] and payload["term"] == self.current_term:
            # Count votes for leadership
            votes_received = sum(1 for member_id in self.cluster_members 
                               if self._has_vote_from(member_id))
            required_votes = len(self.cluster_members) // 2 + 1
            
            if votes_received >= required_votes:
                self._become_leader()
    
    def _handle_append_entries(self, message: CoordinationMessage):
        """Handle append entries (consensus proposal)"""
        payload = message.payload
        
        # Update term if necessary
        if payload["term"] > self.current_term:
            self.current_term = payload["term"]
            self.current_role = AgentRole.FOLLOWER
            self.voted_for = None
        
        # Accept leader
        if payload["term"] == self.current_term:
            self.leader_id = payload["leader_id"]
            self.current_role = AgentRole.FOLLOWER
            
            # Process proposal
            success = self._process_consensus_proposal(payload)
            
            # Send response
            self.send_coordination_message(
                target_agent_id=message.source_agent_id,
                protocol_type=ProtocolType.CONSENSUS,
                message_type="append_response",
                payload={
                    "term": self.current_term,
                    "success": success,
                    "proposal_id": payload.get("proposal_id"),
                    "follower_id": self.agent_id
                },
                priority=8
            )
    
    def _handle_append_response(self, message: CoordinationMessage):
        """Handle append entries response"""
        if self.current_role != AgentRole.LEADER:
            return
        
        payload = message.payload
        proposal_id = payload.get("proposal_id")
        
        if proposal_id in self.pending_proposals:
            proposal = self.pending_proposals[proposal_id]
            
            if payload["success"]:
                proposal.received_votes.add(message.source_agent_id)
                
                # Check if consensus reached
                if len(proposal.received_votes) >= proposal.required_votes:
                    proposal.status = "committed"
                    self._commit_consensus_proposal(proposal)
    
    # Additional message handlers for other protocols
    def _handle_election_start(self, message: CoordinationMessage):
        """Handle election start notification"""
        pass
    
    def _handle_candidate_announcement(self, message: CoordinationMessage):
        """Handle candidate announcement"""
        pass
    
    def _handle_leader_announcement(self, message: CoordinationMessage):
        """Handle leader announcement"""
        payload = message.payload
        
        if payload["term"] >= self.current_term:
            self.current_term = payload["term"]
            self.leader_id = payload["leader_id"]
            self.current_role = AgentRole.FOLLOWER
            self.voted_for = None
    
    def _handle_load_update(self, message: CoordinationMessage):
        """Handle load metrics update"""
        payload = message.payload
        agent_id = message.source_agent_id
        self.agent_loads[agent_id] = payload["load_metrics"]
    
    def _handle_routing_update(self, message: CoordinationMessage):
        """Handle routing table update"""
        pass
    
    def _handle_health_check(self, message: CoordinationMessage):
        """Handle health check request"""
        # Send health check response
        self.send_coordination_message(
            target_agent_id=message.source_agent_id,
            protocol_type=ProtocolType.LOAD_BALANCING,
            message_type="health_check_response",
            payload={
                "agent_id": self.agent_id,
                "status": "healthy",
                "load_metrics": self.get_current_load_metrics(),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _handle_failover_start(self, message: CoordinationMessage):
        """Handle failover start notification"""
        pass
    
    def _handle_failover_complete(self, message: CoordinationMessage):
        """Handle failover completion notification"""
        pass
    
    def _handle_recovery_notification(self, message: CoordinationMessage):
        """Handle agent recovery notification"""
        payload = message.payload
        agent_id = payload["agent_id"]
        
        if agent_id in self.cluster_members:
            self.cluster_members[agent_id].status = "active"
            self.cluster_members[agent_id].last_heartbeat = datetime.now()
            self.logger.info(f"Agent recovered: {agent_id}")
    
    def _handle_state_sync_request(self, message: CoordinationMessage):
        """Handle state synchronization request"""
        pass
    
    def _handle_state_sync_response(self, message: CoordinationMessage):
        """Handle state synchronization response"""
        pass
    
    def _handle_checkpoint_sync(self, message: CoordinationMessage):
        """Handle checkpoint synchronization"""
        pass
    
    def _handle_heartbeat(self, message: CoordinationMessage):
        """Handle heartbeat message"""
        agent_id = message.source_agent_id
        
        if agent_id in self.cluster_members:
            self.cluster_members[agent_id].last_heartbeat = datetime.now()
            
            # Send heartbeat response
            self.send_coordination_message(
                target_agent_id=agent_id,
                protocol_type=ProtocolType.HEARTBEAT,
                message_type="heartbeat_response",
                payload={
                    "responder_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def _handle_heartbeat_response(self, message: CoordinationMessage):
        """Handle heartbeat response"""
        agent_id = message.source_agent_id
        
        if agent_id in self.cluster_members:
            self.cluster_members[agent_id].last_heartbeat = datetime.now()
    
    def _handle_agent_discovery(self, message: CoordinationMessage):
        """Handle agent discovery message"""
        payload = message.payload
        
        agent_id = payload["agent_id"]
        agent_type = payload["agent_type"]
        capabilities = set(payload["capabilities"])
        
        if agent_id not in self.cluster_members:
            # Register new agent
            agent_state = AgentState(
                agent_id=agent_id,
                agent_type=agent_type,
                role=AgentRole.FOLLOWER,
                status="active",
                last_heartbeat=datetime.now(),
                current_term=0,
                voted_for=None,
                log_index=0,
                capabilities=capabilities,
                load_metrics={},
                security_credentials={},
                protocol_participation=set()
            )
            
            self.cluster_members[agent_id] = agent_state
            self.logger.info(f"Discovered new agent: {agent_id}")
    
    def _handle_capability_announcement(self, message: CoordinationMessage):
        """Handle capability announcement"""
        pass
    
    def _handle_topology_update(self, message: CoordinationMessage):
        """Handle network topology update"""
        pass
    
    def _handle_auth_request(self, message: CoordinationMessage):
        """Handle authentication request"""
        pass
    
    def _handle_auth_response(self, message: CoordinationMessage):
        """Handle authentication response"""
        pass
    
    def _handle_key_exchange(self, message: CoordinationMessage):
        """Handle cryptographic key exchange"""
        pass
    
    # Helper methods
    def get_agent_capabilities(self) -> Set[str]:
        """Get capabilities of this agent"""
        return {
            "ml_intelligence", "workflow_orchestration", "coordination",
            "consensus", "load_balancing", "security", "monitoring"
        }
    
    def get_current_load_metrics(self) -> Dict[str, float]:
        """Get current load metrics for this agent"""
        return {
            "cpu_usage": 45.0,  # Simulated
            "memory_usage": 60.0,
            "active_connections": 10,
            "avg_response_time": 250.0,
            "queue_depth": 5
        }
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status"""
        
        active_members = len([m for m in self.cluster_members.values() if m.status == "active"])
        pending_proposals = len([p for p in self.pending_proposals.values() if p.status == "proposed"])
        
        return {
            "agent_info": {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "current_role": self.current_role.value,
                "current_term": self.current_term,
                "leader_id": self.leader_id,
                "voted_for": self.voted_for
            },
            "cluster_status": {
                "total_members": len(self.cluster_members),
                "active_members": active_members,
                "failed_members": len(self.cluster_members) - active_members,
                "cluster_health": active_members / len(self.cluster_members) if self.cluster_members else 0
            },
            "protocol_status": {
                "active_protocols": list(self.active_protocols),
                "pending_proposals": pending_proposals,
                "message_queue_size": len(self.message_queue),
                "load_balancing_rules": len(self.load_balancing_rules)
            },
            "security_status": {
                "security_level": self.protocol_config["security"]["level"].value,
                "trusted_agents": len(self.trusted_agents),
                "auth_tokens_active": len(self.authentication_tokens),
                "encryption_enabled": "encryption" in self.security_keys
            },
            "performance_metrics": {
                "messages_processed": self.sequence_counter,
                "consensus_success_rate": self._calculate_consensus_success_rate(),
                "average_message_latency": 50.0,  # Simulated
                "protocol_efficiency": 0.95
            }
        }
    
    def _calculate_consensus_success_rate(self) -> float:
        """Calculate consensus success rate"""
        if not self.pending_proposals:
            return 1.0
        
        successful = len([p for p in self.pending_proposals.values() if p.status == "committed"])
        total = len(self.pending_proposals)
        
        return successful / total if total > 0 else 1.0
    
    # Security helper methods
    def _sign_message(self, message: CoordinationMessage) -> str:
        """Sign a message for authentication"""
        message_data = f"{message.message_id}:{message.source_agent_id}:{message.timestamp}"
        signature = hmac.new(
            self.security_keys['signing'].encode(),
            message_data.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _verify_message_signature(self, message: CoordinationMessage) -> bool:
        """Verify message signature"""
        if not message.signature:
            return self.protocol_config["security"]["level"] == SecurityLevel.NONE
        
        expected_signature = self._sign_message(message)
        return hmac.compare_digest(expected_signature, message.signature)
    
    def _encrypt_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt message payload"""
        payload_json = json.dumps(payload)
        encrypted_data = self.security_keys['encryption'].encrypt(payload_json.encode())
        return {"encrypted": base64.b64encode(encrypted_data).decode()}
    
    def _decrypt_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt message payload"""
        if "encrypted" not in payload:
            return payload
        
        encrypted_data = base64.b64decode(payload["encrypted"])
        decrypted_data = self.security_keys['encryption'].decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
    
    def _verify_auth_token(self, token: str) -> bool:
        """Verify JWT authentication token"""
        try:
            payload = jwt.decode(token, self.security_keys['signing'], algorithms=['HS256'])
            
            # Check expiration
            if datetime.now().timestamp() > payload.get('expires_at', 0):
                return False
            
            return True
        except jwt.InvalidTokenError:
            return False
    
    def _refresh_auth_tokens(self):
        """Refresh authentication tokens"""
        # Refresh own token if close to expiry
        current_token = self.authentication_tokens.get(self.agent_id)
        if current_token:
            try:
                payload = jwt.decode(current_token, self.security_keys['signing'], algorithms=['HS256'])
                expires_at = payload.get('expires_at', 0)
                
                # Refresh if expires within 1 hour
                if datetime.now().timestamp() + 3600 > expires_at:
                    self.authentication_tokens[self.agent_id] = self._generate_auth_token()
                    self.logger.info("Refreshed authentication token")
            except jwt.InvalidTokenError:
                self.authentication_tokens[self.agent_id] = self._generate_auth_token()
    
    # Protocol implementation helper methods
    def _become_leader(self):
        """Transition to leader role"""
        self.current_role = AgentRole.LEADER
        self.leader_id = self.agent_id
        
        # Announce leadership
        for member_id in self.cluster_members:
            if member_id != self.agent_id:
                self.send_coordination_message(
                    target_agent_id=member_id,
                    protocol_type=ProtocolType.LEADER_ELECTION,
                    message_type="leader_announcement",
                    payload={
                        "leader_id": self.agent_id,
                        "term": self.current_term
                    },
                    priority=9
                )
        
        self.logger.info(f"Became leader for term {self.current_term}")
    
    def _has_vote_from(self, agent_id: str) -> bool:
        """Check if we have a vote from specific agent"""
        # Simplified - in production, track votes properly
        return agent_id in self.cluster_members
    
    def _send_heartbeats(self):
        """Send heartbeats as leader"""
        # Heartbeats are sent in the heartbeat loop
        pass
    
    def _process_pending_proposals(self):
        """Process pending consensus proposals"""
        for proposal in list(self.pending_proposals.values()):
            if proposal.status == "proposed":
                # Check if proposal has timed out
                age = (datetime.now() - proposal.timestamp).total_seconds()
                if age > 300:  # 5 minute timeout
                    proposal.status = "rejected"
                    self.logger.warning(f"Proposal timed out: {proposal.proposal_id}")
    
    def _check_election_timeout(self):
        """Check if election has timed out"""
        # Simplified election timeout check
        pass
    
    def _check_leader_timeout(self):
        """Check if leader has timed out"""
        if self.leader_id and self.leader_id in self.cluster_members:
            leader_state = self.cluster_members[self.leader_id]
            timeout = timedelta(seconds=30)
            
            if datetime.now() - leader_state.last_heartbeat > timeout:
                self.logger.warning("Leader timeout detected, starting election")
                self.start_leader_election()
    
    def _process_consensus_proposal(self, payload: Dict[str, Any]) -> bool:
        """Process a consensus proposal"""
        # Simulate proposal processing
        return True
    
    def _commit_consensus_proposal(self, proposal: ConsensusProposal):
        """Commit a consensus proposal"""
        self.logger.info(f"Committed consensus proposal: {proposal.proposal_type}")
        
        # Remove from pending
        if proposal.proposal_id in self.pending_proposals:
            del self.pending_proposals[proposal.proposal_id]
    
    def _handle_agent_failure(self, agent_id: str):
        """Handle agent failure"""
        self.logger.warning(f"Handling failure of agent: {agent_id}")
        
        # Remove from trusted agents
        self.trusted_agents.discard(agent_id)
        
        # Update load balancing rules
        for rule in self.load_balancing_rules.values():
            if agent_id in rule.target_agents and rule.failover_enabled:
                rule.target_agents.remove(agent_id)
    
    def _rule_matches_requirements(self, rule: LoadBalancingRule, 
                                  requirements: Dict[str, Any]) -> bool:
        """Check if a load balancing rule matches task requirements"""
        # Simplified matching logic
        return bool(rule.target_agents)
    
    def _update_routing_table(self, rule: LoadBalancingRule):
        """Update routing table with new rule"""
        self.routing_table[rule.rule_id] = rule
    
    def _update_all_routing_tables(self):
        """Update all routing tables"""
        # Simplified routing table updates
        pass
    
    def _check_failover_conditions(self):
        """Check if failover is needed"""
        # Simplified failover condition checking
        pass
    
    def _collect_load_metrics(self):
        """Collect current load metrics"""
        self.agent_loads[self.agent_id] = self.get_current_load_metrics()
    
    def _detect_security_anomalies(self):
        """Detect security anomalies"""
        # Simplified security anomaly detection
        pass
    
    def _update_trusted_agents(self):
        """Update list of trusted agents"""
        # Remove agents that haven't been active
        current_time = datetime.now()
        timeout = timedelta(hours=24)
        
        inactive_agents = set()
        for agent_id in self.trusted_agents:
            if (agent_id in self.cluster_members and
                current_time - self.cluster_members[agent_id].last_heartbeat > timeout):
                inactive_agents.add(agent_id)
        
        self.trusted_agents -= inactive_agents
    
    def _optimize_protocol_parameters(self):
        """Optimize protocol parameters based on performance"""
        # Simplified protocol optimization
        pass
    
    def _analyze_protocol_performance(self):
        """Analyze protocol performance metrics"""
        # Simplified performance analysis
        pass
    
    def stop_coordination(self):
        """Stop coordination protocols"""
        self.coordination_active = False
        self.logger.info("Agent coordination protocols stopped")

def main():
    """Main function for standalone execution"""
    coordinator = AgentCoordinationProtocols("ml_agent_001", "ml_intelligence")
    
    try:
        # Simulate some coordination activities
        coordinator.register_agent(
            agent_id="test_agent_001",
            agent_type="testing",
            capabilities={"test_generation", "validation"},
            endpoint="http://localhost:8003"
        )
        
        coordinator.create_load_balancing_rule(
            name="ML Processing Load Balancing",
            algorithm="weighted",
            target_agents=["ml_agent_001", "test_agent_001"],
            weights={"ml_agent_001": 0.7, "test_agent_001": 0.3}
        )
        
        # Monitor status
        while True:
            status = coordinator.get_coordination_status()
            print(f"\n{'='*80}")
            print("AGENT COORDINATION PROTOCOLS STATUS")
            print(f"{'='*80}")
            print(f"Agent Role: {status['agent_info']['current_role']}")
            print(f"Cluster Members: {status['cluster_status']['total_members']}")
            print(f"Active Protocols: {status['protocol_status']['active_protocols']}")
            print(f"Security Level: {status['security_status']['security_level']}")
            print(f"Consensus Success: {status['performance_metrics']['consensus_success_rate']:.1%}")
            print(f"{'='*80}")
            
            time.sleep(60)  # Status update every minute
            
    except KeyboardInterrupt:
        coordinator.stop_coordination()
        print("\nAgent coordination stopped.")

if __name__ == "__main__":
    main()