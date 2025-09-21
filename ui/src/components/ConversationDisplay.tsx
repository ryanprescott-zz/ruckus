import React from 'react';
import type { LLMConversationResult, PromptMessage, PromptRole } from '../types/api';
import './ConversationDisplay.css';

interface ConversationDisplayProps {
  conversationResult: LLMConversationResult;
  title?: string;
}

const ConversationDisplay: React.FC<ConversationDisplayProps> = ({
  conversationResult,
  title = "LLM Conversation"
}) => {
  const { conversation, input_tokens, output_tokens, total_tokens } = conversationResult;

  const getRoleIcon = (role: PromptRole): string => {
    switch (role) {
      case 'system': return '⚙️';
      case 'user': return '👤';
      case 'assistant': return '🤖';
      default: return '💬';
    }
  };

  const getRoleLabel = (role: PromptRole): string => {
    switch (role) {
      case 'system': return 'System';
      case 'user': return 'User';
      case 'assistant': return 'Assistant';
      default: return role;
    }
  };

  const getRoleClassName = (role: PromptRole): string => {
    return `message-${role}`;
  };

  if (!conversation || conversation.length === 0) {
    return (
      <div className="conversation-container">
        <div className="conversation-header">
          <h3>{title}</h3>
        </div>
        <div className="no-conversation">
          <p>No conversation data available</p>
          <p className="hint">The model response may not have been captured.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="conversation-container">
      <div className="conversation-header">
        <div className="conversation-title">
          <h3>{title}</h3>
          {total_tokens && (
            <span className="token-count">
              {total_tokens} tokens ({input_tokens} in, {output_tokens} out)
            </span>
          )}
        </div>
      </div>

      <div className="conversation-messages">
        {conversation.map((message, index) => (
          <div
            key={index}
            className={`message ${getRoleClassName(message.role)}`}
          >
            <div className="message-header">
              <span className="message-role">
                {getRoleIcon(message.role)} {getRoleLabel(message.role)}
              </span>
            </div>
            <div className="message-content">
              {message.content}
            </div>
          </div>
        ))}
      </div>

      <div className="conversation-footer">
        <div className="conversation-stats">
          <span className="stat-item">
            Messages: {conversation.length}
          </span>
          {input_tokens && (
            <span className="stat-item">
              Input: {input_tokens} tokens
            </span>
          )}
          {output_tokens && (
            <span className="stat-item">
              Output: {output_tokens} tokens
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default ConversationDisplay;