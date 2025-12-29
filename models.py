# import datetime
# from sqlalchemy import (
#     String,
#     Integer,
#     DateTime,
#     Boolean,
#     BigInteger,
#     ForeignKey,
#     Text,
# )
# from sqlalchemy.orm import (
#     DeclarativeBase,
#     Mapped,
#     mapped_column,
#     relationship,
# )
# from sqlalchemy.sql import func

# from sqlalchemy.dialects.postgresql import JSONB
# # ============================================================
# # Base
# # ============================================================

# class Base(DeclarativeBase):
#     pass



# import datetime
# from sqlalchemy import (
#     Column,
#     String,
#     DateTime,
#     ForeignKey,
#     Index,
#     Text,
# )
# from sqlalchemy.orm import relationship
# from sqlalchemy.sql import func

# class Message(Base):
#     __tablename__ = "Message"

#     id = Column(
#         String,
#         primary_key=True,
#         default=lambda: str(__import__("uuid").uuid4()),
#     )

#     sessionId = Column(
#         String,
#         ForeignKey("Session.sessionId", ondelete="CASCADE"),
#         nullable=False,
#     )

#     role = Column(String, nullable=False)
#     text = Column(Text, nullable=False)

#     createdAt = Column(
#         DateTime(timezone=True),
#         server_default=func.now(),
#         nullable=False,
#     )

#     # 🔗 Relation
#     session = relationship(
#         "Session",
#         back_populates="messages",
#     )

#     __table_args__ = (
#         Index("ix_message_session_created", "sessionId", "createdAt"),
#     )

   

# # ============================================================
# # Company
# # ============================================================

# class Company(Base):
#     __tablename__ = "Company"

#     id: Mapped[str] = mapped_column(String, primary_key=True)
#     name: Mapped[str] = mapped_column(String)

#     timeZone: Mapped[str | None] = mapped_column(String)
#     businessHoursStart: Mapped[int | None] = mapped_column(Integer)
#     businessHoursEnd: Mapped[int | None] = mapped_column(Integer)

#     createdAt: Mapped[datetime.datetime] = mapped_column(
#         DateTime, server_default=func.now()
#     )
#     updatedAt: Mapped[datetime.datetime | None] = mapped_column(
#         DateTime, onupdate=func.now()
#     )

#     sessionCounter: Mapped[int] = mapped_column(Integer)

#     # Relations
#     bots = relationship("Bot", back_populates="company")
#     users = relationship("User", back_populates="company")
#     sessions = relationship("Session", back_populates="company")


# # ============================================================
# # Bot
# # ============================================================

# class Bot(Base):
#     __tablename__ = "Bot"

#     id: Mapped[str] = mapped_column(String, primary_key=True)
#     name: Mapped[str] = mapped_column(String)

#     publicApiKey: Mapped[str] = mapped_column(String)
#     botName: Mapped[str | None] = mapped_column(String)
#     welcomeMessage: Mapped[str | None] = mapped_column(Text)
#     systemInstruction: Mapped[str | None] = mapped_column(Text)
#     widgetColor: Mapped[str | None] = mapped_column(String)
#     botLogoUrl: Mapped[str | None] = mapped_column(String)
#     popupDelay: Mapped[int | None] = mapped_column(Integer)

#     companyId: Mapped[str] = mapped_column(
#         String,
#         ForeignKey("Company.id", ondelete="CASCADE"),
#         index=True,
#     )

#     company = relationship("Company", back_populates="bots")
#     knowledgeSources = relationship("KnowledgeSource", back_populates="bot")
#     sessions = relationship("Session", back_populates="bot")
#     bookings = relationship("Booking", back_populates="bot")


# # ============================================================
# # KnowledgeSource
# # ============================================================

# class KnowledgeSource(Base):
#     __tablename__ = "KnowledgeSource"

#     id: Mapped[str] = mapped_column(String, primary_key=True)
#     fileName: Mapped[str] = mapped_column(String)
#     storagePath: Mapped[str] = mapped_column(String, nullable=False)
#     fileType: Mapped[str] = mapped_column(String)

#     createdAt: Mapped[datetime.datetime] = mapped_column(
#         DateTime, server_default=func.now()
#     )

#     botId: Mapped[str] = mapped_column(
#         String,
#         ForeignKey("Bot.id", ondelete="CASCADE"),
#         index=True,
#     )

#     bot = relationship("Bot", back_populates="knowledgeSources")
#     chunks = relationship("KnowledgeChunk", back_populates="knowledgeSource")


# # ============================================================
# # KnowledgeChunk
# # NOTE: embedding column intentionally OMITTED (Pinecone)
# # ============================================================

# class KnowledgeChunk(Base):
#     __tablename__ = "KnowledgeChunk"

#     id: Mapped[str] = mapped_column(String, primary_key=True)
#     content: Mapped[str] = mapped_column(Text)

#     createdAt: Mapped[datetime.datetime] = mapped_column(
#         DateTime, server_default=func.now()
#     )
#     updatedAt: Mapped[datetime.datetime | None] = mapped_column(
#         DateTime, onupdate=func.now()
#     )

#     knowledgeSourceId: Mapped[str] = mapped_column(
#         String,
#         ForeignKey("KnowledgeSource.id", ondelete="CASCADE"),
#         index=True,
#     )

#     knowledgeSource = relationship("KnowledgeSource", back_populates="chunks")


# # ============================================================
# # Session
# # ============================================================

# # class Session(Base):
# #     __tablename__ = "Session"

# #     sessionId: Mapped[str] = mapped_column(String, primary_key=True)

# #     status: Mapped[str] = mapped_column(String)
# #     chatStatus: Mapped[str] = mapped_column(String)

# #     lastSeen: Mapped[int] = mapped_column(BigInteger)
# #     lastMessage: Mapped[str] = mapped_column(Text)

# #     ip: Mapped[str] = mapped_column(String)
# #     userAgent: Mapped[str | None] = mapped_column(String)
# #     location: Mapped[str | None] = mapped_column(String)

# #     isOnline: Mapped[bool] = mapped_column(Boolean)
# #     requiresAttention: Mapped[bool] = mapped_column(Boolean)

# #     lastMessageAt: Mapped[datetime.datetime | None] = mapped_column(DateTime)

# #     companyId: Mapped[str] = mapped_column(
# #         String, ForeignKey("Company.id", ondelete="CASCADE"), index=True
# #     )

# #     botId: Mapped[str] = mapped_column(
# #         String, ForeignKey("Bot.id", ondelete="CASCADE"), index=True
# #     )

# #     company = relationship("Company", back_populates="sessions")
# #     bot = relationship("Bot", back_populates="sessions")
# #     messages = relationship(
# #         "Message",
# #         back_populates="session",
# #         cascade="all, delete-orphan",
# #         order_by="Message.createdAt",
# #     )

# # ============================================================
# # Booking
# # ============================================================

# class Booking(Base):
#     __tablename__ = "Booking"

#     id: Mapped[str] = mapped_column(String, primary_key=True)
#     date: Mapped[datetime.datetime] = mapped_column(DateTime)
#     name: Mapped[str] = mapped_column(String)
#     email: Mapped[str] = mapped_column(String)
#     phone: Mapped[str] = mapped_column(String)
#     details: Mapped[str] = mapped_column(Text)

#     botId: Mapped[str] = mapped_column(
#         String, ForeignKey("Bot.id", ondelete="CASCADE"), index=True
#     )

#     bot = relationship("Bot", back_populates="bookings")


# # ============================================================
# # User
# # ============================================================

# class User(Base):
#     __tablename__ = "User"

#     id: Mapped[str] = mapped_column(String, primary_key=True)
#     email: Mapped[str] = mapped_column(String)
#     passwordHash: Mapped[str] = mapped_column(String)

#     createdAt: Mapped[datetime.datetime] = mapped_column(
#         DateTime, server_default=func.now()
#     )
#     updatedAt: Mapped[datetime.datetime | None] = mapped_column(
#         DateTime, onupdate=func.now()
#     )

#     role: Mapped[str] = mapped_column(String)
#     companyId: Mapped[str] = mapped_column(
#         String, ForeignKey("Company.id", ondelete="CASCADE"), index=True
#     )

#     company = relationship("Company", back_populates="users")




# class Session(Base):
#     __tablename__ = "Session"

#     # =========================
#     # Primary Key
#     # =========================
#     sessionId = Column(String, primary_key=True)

#     # =========================
#     # Core Status
#     # =========================
#     status = Column(String, nullable=False)
#     chatStatus = Column(String, nullable=False, default="GREEN")

#     # =========================
#     # Activity Tracking
#     # =========================
#     lastSeen = Column(BigInteger, nullable=False)
#     lastMessage = Column(String, nullable=False)
#     lastMessageAt = Column(DateTime(timezone=True), server_default=func.now())

#     # =========================
#     # Client Info
#     # =========================
#     ip = Column(String, nullable=False)
#     userAgent = Column(String, nullable=True)
#     location = Column(String, nullable=True)

#     # =========================
#     # Conversation State
#     # =========================
#     currentNodeId = Column(String, nullable=True)
#     variables = Column(JSONB, nullable=True, server_default="{}")

#     # =========================
#     # Flags
#     # =========================
#     isOnline = Column(Boolean, nullable=False, default=False)
#     requiresAttention = Column(Boolean, nullable=False, default=False)
#     isToReassign = Column(Boolean, nullable=False, default=False)
#     wasEverAssignedToAdmin = Column(Boolean, nullable=False, default=False)

#     # =========================
#     # Assignment
#     # =========================
#     assignedToId = Column(
#         String,
#         ForeignKey("User.id", ondelete="SET NULL"),
#         nullable=True,
#     )

#     lastAssignedTo = Column(JSONB, nullable=True)

#     # =========================
#     # Optional
#     # =========================
#     sessionNumber = Column(Integer, nullable=True)
#     privateNotes = Column(JSONB, nullable=True, server_default="[]")

#     # =========================
#     # Foreign Keys
#     # =========================
#     companyId = Column(
#         String,
#         ForeignKey("Company.id", ondelete="CASCADE"),
#         nullable=False,
#     )

#     botId = Column(
#         String,
#         ForeignKey("Bot.id", ondelete="CASCADE"),
#         nullable=False,
#     )

#     # =========================
#     # Relationships
#     # =========================
#     messages = relationship(
#         "Message",
#         back_populates="session",
#         cascade="all, delete-orphan",
#         order_by="Message.createdAt",
#     )

#     company = relationship(
#         "Company",
#         back_populates="sessions",
#     )

#     bot = relationship(
#         "Bot",
#         back_populates="sessions",
#     )

#     assignedTo = relationship(
#         "User",
#         foreign_keys=[assignedToId],
#     )

#     # =========================
#     # Indexes
#     # =========================
#     __table_args__ = (
#         Index("ix_session_assignedToId", "assignedToId"),
#         Index("ix_session_company_attention", "companyId", "requiresAttention"),
#         Index("ix_session_botId", "botId"),
#     )